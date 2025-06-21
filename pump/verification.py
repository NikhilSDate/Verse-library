from artificial_pancreas_simulate import *
from verse_model import *
from artificial_pancreas_agent import *
from artificial_pancreas_scenario import *
from pump_model import *
from cgm import *
from simutils import *
from hovorka_model import patient_original
import pickle
import random
from pyrsistent import freeze, thaw
from dataclasses import asdict
import yaml
import argparse
from shutil import rmtree
from simutils import FORGOT_BOLUS
from safety.safety import realism
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import signal
import itertools
import ast
from tqdm import tqdm
from typing import Any


# TODO: this function is a bit of a hack
def denumpify(obj):
    if isinstance(obj, dict):
        return {k: denumpify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [denumpify(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(denumpify(i) for i in obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return denumpify(obj.tolist())  # recursively convert the list too
    else:
        return obj
    

def custom_asdict_factory(data):
    def convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        return obj
    return dict((k, convert_value(v)) for k, v in data)

def get_allowed_meal_carb_ranges(TOTAL_LOW, TOTAL_HIGH, num_meals=4):
    meal_carb_ranges = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150)]
    m = len(meal_carb_ranges)
    good_ranges = []
    for i in range(m ** num_meals):
        n = i
        idx0 = n % m
        n  = n // m
        idx1 = n % m
        n = n // m
        idx2 = n % m
        n = n // m
        idx3 = n % m
        assert(n // m == 0)
        ranges = [meal_carb_ranges[idx0], meal_carb_ranges[idx1], meal_carb_ranges[idx2], meal_carb_ranges[idx3]]
        low_sum = sum([ranges[i][0] for i in range(4)])
        high_sum = sum([ranges[i][1] for i in range(4)])

        # accept any range that has at least partial overlap with the total range
        if low_sum >= TOTAL_HIGH or high_sum <= TOTAL_LOW:
            continue
        good_ranges.append(ranges)
    return good_ranges
    
def gen_verification_scenarios():
    # we want a set of conditions that a scenario should satisfy to ensure realism
    # M1: total carbs should be in a particular range
    # M2: carbs in each meal should be in a particular range
    # M3: number of meals should be reasonable (let's say 4 meals: (6 AM to 11 AM, 11 AM to 3 PM, 3 PM to 7 PM, 7 PM TO 11 PM))
    # M4: meals should be separated by at least 30 minutes
    # M7: each meal is either a low fat/carb meal, or a high carb
    
    # maybe
    # M7: carbs in each meal should be >= A% (10%) of total carbs, and <= B% of total carbs (70%)
    
    # B1: user always correctly selects extended bolus settings for high fat/carb meal (maybe give one or two options)
    # B2: user boluses in [-20, 20] of meal time (should this be discrete or continuous, should we also consider forgotten boluses)
    # B3: user makes up to a 10% error in carb amount
    # B4: user always requests a correction and enters BG exactly as it appears on the CGM
    
    # CGM1: CGM value is always at a constant offset from true value (let's say offset in [-20, 20], other option would be to sample the error parameters from the model, start at a random day offset, and set all the error terms to 0)
    
    # G1: starting BG is in the normal range (70, 180)
    
    # for now take this and run it through the AGP report
        
    # source: https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/carbohydrates/art-20045705

    # TODO: CGM errors

    DURATION = 24 * 60

    ERROR_LOW = 0.9
    ERROR_HIGH = 1.1
        
    PATIENT_BASAL_GLUCOSE = 6.5
    CGM_BIAS = 0.1
    NUM_MEALS = 4
    
    BOLUS_OFFSET = -5
    
    BASAL_RATE_RANGE = 0.1

    RESUME = True
    
    meal_ranges = get_allowed_meal_carb_ranges(100, 350)
    
    meal_1_time = 60 * np.array([2, 5])
    meal_2_time = 60 * np.array([7, 10])
    meal_3_time = 60 * np.array([11, 13])
    meal_4_time = 60 * np.array([14, 17])
    
    meal_times_lists = [meal_1_time, meal_2_time, meal_3_time, meal_4_time]
    meal_times = list(itertools.product(*meal_times_lists))
    
    taum_choices = [DEFAULT_MEAL, HIGH_FAT_MEAL]
    taum_default = [DEFAULT_MEAL]

    meal_TauM_lists = [taum_choices, taum_choices, taum_default, taum_choices]
    meal_TauMs = list(itertools.product(*meal_TauM_lists))
    
    meal_params = itertools.product(*[meal_times, meal_ranges, meal_TauMs])
    
    meals_choices: List[List[Meal]] = []
    for comb in meal_params:
        times = comb[0]
        carbs = comb[1]
        TauMs = comb[2]
        scenario_meals = []
        for i in range(NUM_MEALS):
            scenario_meals.append(Meal(times[i], carbs[i], TauMs[i]))
        meals_choices.append(scenario_meals)    
    
    bolus_configs = {DEFAULT_MEAL: (BolusType.Simple, None), HIGH_FAT_MEAL: (BolusType.Extended, ExtendedBolusConfig(50, 180))}    
    
    scenarios = []
    for meals in meals_choices:  
        
        boluses = []

        for i, m in enumerate(meals):
            bolus = Bolus(m.time + BOLUS_OFFSET, None, None, i, True, None)
            bolus = set_bolus_config(bolus, bolus_configs[m.TauM])
            boluses.append(bolus)
                
        errors = [ERROR_LOW, ERROR_HIGH]
        init_bg = [70, 180]
        patient_params = patient_original({'basalGlucose': PATIENT_BASAL_GLUCOSE})
        settings = get_recommended_settings(TDD=39.22, BW=74.9)
        settings['basal_rate'] = patient_params['Ub']
        settings['basal_iq'] = True
        
        settings_low = settings.copy()
        settings_high = settings.copy()
        
        settings_low['basal_rate'] = settings_low['basal_rate'] * (1 - BASAL_RATE_RANGE)
        settings_high['basal_rate'] = settings_high['basal_rate'] * (1 + BASAL_RATE_RANGE)
        
        cgm_config = CGMConfig((1 - CGM_BIAS, 1 + CGM_BIAS), (0, 0))
        user_config = UserConfig(resume=RESUME)
        scenario = SimulationScenario(init_bg, boluses, meals, errors, [settings_low, settings_high], patient_params, cgm_config, sim_duration=DURATION, user_config=user_config)
        scenarios.append(scenario)
    return scenarios        
        
def get_scenario_directory(scenario: SimulationScenario, output_dir):
    idx = 0
    h = hex(hash(scenario) + sys.maxsize + 1)[2:]
    result = ''
    while True:
        prefix = hex(idx)[2:]
        attempt = os.path.join(output_dir, f'scenario_{prefix}{h}')
        if os.path.exists(attempt):
            with open(os.path.join(attempt, 'scenario.pkl'), 'rb') as f:
                collision = pickle.load(attempt)
            if collision == scenario:
                return None
            else:
                idx += 1
        else:
            result = attempt
            break
    os.makedirs(result)
    return result

def save_scenario_results(scenario: SimulationScenario, traces, safety_results, output_dir):
    # create a directory in output_dir using hash of scenario
    scenario_directory = get_scenario_directory(scenario, output_dir)
    if scenario_directory is None:
        print('redundant scenario')
        return
    fig = plot_variable(traces, 'G', show=False)
    with open(os.path.join(scenario_directory, 'traces.pkl'), 'wb') as f:
        pickle.dump(traces, f)
    fig.write_image(os.path.join(scenario_directory, 'plot.png'))
    with open(os.path.join(scenario_directory, 'safety.txt'), 'w') as f:
        f.write(str(safety_results))
    with open(os.path.join(scenario_directory, 'scenario.yaml'), 'w') as f:
        to_dump = denumpify(asdict(scenario.get_data(), dict_factory=custom_asdict_factory))
        yaml.dump(to_dump, f)
    with open(os.path.join(scenario_directory, 'scenario.pkl'), 'wb') as f:
        pickle.dump(scenario, f)

def save_crash(scenario, payload, output_dir):
    scenario_directory = get_scenario_directory(scenario, output_dir)
    payload.save(scenario_directory)

def scenario_exists(scenario, output_dir):


def run_verification_scenario(scenario, output_dir):
    if scenario_exists(output_dir):
        pass

    res = verify_multi_meal_scenario(scenario)
    if res.type == ResultType.OK:
        traces = res.payload
        safety_results = evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))
        save_scenario_results(scenario, traces, safety_results, output_dir)
    else:
        save_crash(scenario, res.payload, output_dir)

def sigint(signum, frame):
    os.kill(0, signal.SIGKILL)

def verify(scenarios: List[SimulationScenario], output_dir: str, pool_size: int):
    with Pool(pool_size) as p:
        run_func = lambda scenario: run_verification_scenario(scenario, output_dir)
        p.map(run_func, scenarios)

# load all results
# there is no point trying to optimize this, since this is not really the bottleneck
def load_results(output_dir) -> List[Tuple[Scenario, object, object]]:
    results = []
    scenario_dirs = [ f for f in os.scandir(output_dir) if f.is_dir() ]
    for scenario_dir in tqdm(scenario_dirs):
        try:
            with open(os.path.join(scenario_dir.path, 'scenario.pkl'), 'rb') as f:
                scenario = pickle.load(f)
            with open(os.path.join(scenario_dir.path, 'traces.pkl'), 'rb') as f:
                traces = pickle.load(f)
            with open(os.path.join(scenario_dir.path, 'safety.txt')) as f:
                safety = ast.literal_eval(f.read())
            results.append((scenario, traces, safety))
        except FileNotFoundError:
            pass
    return results  

def load_from_dir(output_dir, scenario_dir) -> Tuple[SimulationScenario, Any, Any]:
    scenario_dir = os.path.join(output_dir, scenario_dir)
    scenario, traces, safety = None, None, None
    scenario_path = os.path.join(scenario_dir, 'scenario.pkl')
    traces_path = os.path.join(scenario_dir, 'traces.pkl')
    safety_path = os.path.join(scenario_dir, 'safety.txt')
    if os.path.exists(scenario_path):
        with open(scenario_path, 'rb') as f:
            scenario = pickle.load(f)
    if os.path.exists(traces_path):
        with open(traces_path, 'rb') as f:
            traces = pickle.load(f)
    if os.path.exists(safety_path):
        with open(safety_path) as f:
            safety = ast.literal_eval(f.read())
    return (scenario, traces, safety)    

def load_from_dir_err(output_dir, scenario_dir) -> Tuple[SimulationScenario, List]:
    scenario_dir = os.path.join(output_dir, scenario_dir)
    scenario, init = None, None
    scenario_path = os.path.join(scenario_dir, 'scenario.pkl')
    init_path = os.path.join(scenario_dir, 'init.pkl')
    if os.path.exists(scenario_path):
        with open(scenario_path, 'rb') as f:
            scenario = pickle.load(f)
    if os.path.exists(init_path):
        with open(init_path, 'rb') as f:
            init = pickle.load(f)
    return scenario, init

def debug_scenario(scenario_path):
    with open(os.path.join(scenario_path, 'scenario.pkl'), 'rb') as f:
        scenario = pickle.load(f)
    signal.signal(signal.SIGINT, sigint)
    traces = run_verification_scenario(scenario, logging=True)
    fig1 = plot_variable(traces, 'G', show=True)
    fig2 = plot_variable(traces, 'InsSub1', show=True)
    fig1.write_image('debug_G.png')
    fig2.write_image('debug_InsSub.png')
            
            
def verify_wrapper():
    parser = argparse.ArgumentParser('pumpverif')
    parser.add_argument('-p', '--processes', default=1, type=int)
    parser.add_argument('-s', '--seed', default=42, type=int)
    parser.add_argument('-o', '--output-dir', default='results/verification', type=str)
    args = parser.parse_args()
    seed = args.seed
    processes = args.processes
    output_dir = args.output_dir
    signal.signal(signal.SIGINT, sigint)
    np.random.seed(seed)
    random.seed(seed)

    scenarios = gen_verification_scenarios()
    np.random.shuffle(scenarios)  
    
    # don't want to redo existing scenarios
    results = load_results(output_dir)
    existing = set(result[0] for result in results)
    print('found {} existing results')
    scenarios = set(scenarios).difference(existing)

    verify(scenarios, output_dir, pool_size=processes)  

def save_perfectly_unsafe(results, output_dir):
    for result in results:
        if np.min(1 - np.array(result[2], dtype=int)) > 0:
            save_scenario_results(result[0], result[1], result[2], output_dir)

if __name__ == '__main__':
    verify_wrapper()