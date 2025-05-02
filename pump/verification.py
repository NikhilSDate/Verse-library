from artificial_pancreas_simulate import *
from verse_model import *
from artificial_pancreas_agent import *
from pump_model import *
from cgm import *
from simutils import *
from hovorka_model import HovorkaModel, patient_original
import pickle
import random
import json
from pyrsistent import freeze, thaw
from dataclasses import asdict
import yaml
import argparse
from shutil import rmtree
from simutils import FORGOT_BOLUS
from safety.safety import realism
import matplotlib.pyplot as plt
from multiprocessing import Pool
import threading
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

def check_scenario(scenario: SimulationScenario):
    # source: https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/carbohydrates/art-20045705
    TOTAL_CARBS_LOW = 100
    TOTAL_CARBS_HIGH = 350
    MEAL_TIME_RANGES = [(60, 360), (360, 600), (600, 840), (840, 1080)]
    INTER_MEAL_TIME = 30
    ALLOWED_TMAX = [DEFAULT_MEAL, HIGH_FAT_MEAL]
    TMAX_TO_CONFIG = {DEFAULT_MEAL: [(BolusType.Simple, None)], HIGH_FAT_MEAL: [(BolusType.Extended, ExtendedBolusConfig(50, 180))]}
    BOLUS_MEAL_DELTA = 20
    
    meals = scenario.get_meals()
    if not (len(meals) == 4):
        print('bad number of meals')
        return False
    
    # M1
    total_carbs_low = 0
    total_carbs_high = 0
    for meal in meals:
        total_carbs_low += meal.carbs[0]
        total_carbs_high += meal.carbs[1]
    if not (total_carbs_low >= TOTAL_CARBS_LOW and total_carbs_high <= TOTAL_CARBS_HIGH):
        print('bad total carbs: ', total_carbs_low, total_carbs_high)
        return False
    
    # M3
    # we will start scenario at 5 AM
    for i in range(len(MEAL_TIME_RANGES)):
        if not (meals[i].time >= MEAL_TIME_RANGES[i][0] and meals[i].time <= MEAL_TIME_RANGES[i][1]):
            print('bad meal times')
            return False
    
    # M4
    for i in range(len(meals) - 1):
        if not (meals[i + 1].time - meals[i].time) >= INTER_MEAL_TIME:
            print('bad inter-meal times')
            return False
    
    for i in range(len(meals)):
        if not (meals[i].TauM in ALLOWED_TMAX):
            print('bad t_max')
            return False
    
    meal_index_to_bolus = scenario.get_bolus_meal_mapping()    
    # B1    
    for i in range(len(meals)):
        bolus = meal_index_to_bolus[i]
        meal = meals[i]
        allowed_configs = TMAX_TO_CONFIG[meal.TauM]
        if not get_bolus_config(bolus) in allowed_configs:
            print(get_bolus_config(bolus), allowed_configs)
            print('bad bolus config')
            return False
    
    for i in range(len(meals)):
        bolus = meal_index_to_bolus[i]
        meal = meals[i]
        if not abs(bolus.time - meal.time) <= BOLUS_MEAL_DELTA:
            print('bad bolus meal delta')
            return False
        
    for bolus in meal_index_to_bolus.values():
        if not bolus.correction:
            print('bad bolus correction')
            return False
        
    return True

def get_allowed_meal_carb_ranges(TOTAL_LOW, TOTAL_HIGH, num_meals=4):
    meal_carb_ranges = [(0, 37.5), (37.5, 75), (75, 112.5), (112.5, 150)]
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
    
    # we might be able to do interesting caching with 
    
    
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
        
def get_scenario_directory(scenario: Scenario, log_dir):
    idx = 0
    h = hex(hash(scenario) + sys.maxsize + 1)[2:]
    result = ''
    while True:
        prefix = hex(idx)[2:]
        attempt = os.path.join(log_dir, f'scenario_{prefix}{h}')
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

def save_scenario_results(scenario: SimulationScenario, traces, safety_results, log_dir):
    # create a directory in log_dir using hash of scenario
    scenario_directory = get_scenario_directory(scenario, log_dir)
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

def run_verification_scenario(scenario, logging=False):
    tqdm.write(str(scenario))
    traces = verify_multi_meal_scenario(scenario, logging=logging)
    safety_results = evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))
    save_scenario_results(scenario, traces, safety_results, 'results/verification')

def sigint(signum, frame):
    os.kill(0, signal.SIGKILL)

def verify(scenarios: List[SimulationScenario], pool_size: int):
    with Pool(pool_size) as p:
        p.map(run_verification_scenario, scenarios)

# load all results
# there is no point trying to optimize this, since this is not really the bottleneck
def load_results(log_dir) -> List[Tuple[Scenario, object, object]]:
    results = []
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() ]
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

def load_from_dir(log_dir, scenario_dir) -> Tuple[SimulationScenario, Any, Any]:
    scenario_dir = os.path.join(log_dir, scenario_dir)
    with open(os.path.join(scenario_dir, 'scenario.pkl'), 'rb') as f:
        scenario = pickle.load(f)
    with open(os.path.join(scenario_dir, 'traces.pkl'), 'rb') as f:
        traces = pickle.load(f)
    with open(os.path.join(scenario_dir, 'safety.txt')) as f:
        safety = ast.literal_eval(f.read())
    return (scenario, traces, safety)    

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
    args = parser.parse_args()
    signal.signal(signal.SIGINT, sigint)
    np.random.seed(42)
    scenarios = gen_verification_scenarios()
    np.random.shuffle(scenarios)  
    verify(scenarios, pool_size=args.processes)  
    
def compute_proof_statistics(results):
    totals = np.zeros_like(results[0][2], dtype=int)
    perfect = 0
    perfectly_unsafe = 0
    for result in tqdm(results):
        totals += np.array(result[2], dtype=int)
        perfect += np.min(np.array(result[2], dtype=int))
        perfectly_unsafe += np.min(1 - np.array(result[2], dtype=int))
    return totals / len(results), perfect / len(results), perfectly_unsafe / len(results)

def save_perfectly_unsafe(results, log_dir):
    for result in results:
        if np.min(1 - np.array(result[2], dtype=int)) > 0:
            save_scenario_results(result[0], result[1], result[2], log_dir)

def unsafe_analysis(results: List[Tuple[Scenario, object, object]], index):
    points_safe = []
    points_unsafe = []

    unsafe_map = {}
    safe_map = {}

    for result in results:
        point = [result[0].get_largest_meal(), result[0].get_total_carb_range()[1]]
        map_idx = 1
        if result[2][index]:
            points_safe.append(point)
            safe_map[point[map_idx]] = safe_map.get(point[map_idx], 0) + 1
        else:
            points_unsafe.append(point)
            unsafe_map[point[map_idx]] = unsafe_map.get(point[map_idx], 0) + 1
    
    print(len(points_safe))
    print(len(points_unsafe))
    points_safe = np.array(points_safe)
    points_unsafe = np.array(points_unsafe)
    plt.scatter(points_safe[:, 0], points_safe[:, 1], c='green', alpha=0.1)
    plt.scatter(points_unsafe[:, 0], points_unsafe[:, 1], c='red', alpha=0.1)
    plt.legend()
    plt.xlabel('Largest meal (g)')
    plt.ylabel('Total carbs upper bound (g)')
    plt.savefig('unsafe.png')
    return safe_map, unsafe_map

def get_init(scenario, index):
    inits = verify_multi_meal_scenario(scenario, track_inits=True)
    return inits[index]

def overlay_simulation_traces(args):

    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
        '#111111',  # off-white
        '#ffffff'   # black
    ]

    (log_dir, scenario_dir) = args
    scenario, verification_traces, safety = load_from_dir(log_dir, scenario_dir)
    fig = plot_variable(verification_traces, 'G', show=False)
    inits = verify_multi_meal_scenario(scenario, track_inits=True)
    for i, init in enumerate(inits):
        traces = simulate_from_init(scenario, init)
        plot_variable(traces, 'G', show=False, fig=fig, color=colors[i])

        # custom legend
        fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"trace {i}",
                marker=dict(size=7, color=colors[i], symbol='square'),
        ))
    y_mins = []
    y_maxs = []
    for trace_data in fig.data:
        y_mins.append(min(trace_data.y))
        y_maxs.append(max(trace_data.y))
    fig.update_layout(showlegend=True)
    fig.write_image(os.path.join(log_dir, scenario_dir, 'plot_with_sims_fixed.png'))
    return fig
    
if __name__ == '__main__':
    # results = load_results('results/verification')
    # save_perfectly_unsafe(results, 'results/perfectly_unsafe')

    # scenario.user_config = UserConfig(resume=True)
    # traces = verify_multi_meal_scenario(scenario)
    # plot_variable(traces, 'G')

    # with open('results/verification/scenario_080000000000d3c9b/traces.pkl', 'rb') as f:
    #     traces = pickle.load(f)
    # plot_variable(traces, 'G')
    # stats = compute_proof_statistics(results)    
    # plot_verification_results(results, 0)
    # results/perfectly_unsafe/scenario_0800000001e04ba8e
    # signal.signal(signal.SIGINT, sigint)    
    # scenario, traces, unsafe = load_from_dir(log_dir, 'scenario_0800000001b086758')
    # perfectly_unsafe_scenarios = [(log_dir, f.name) for f in os.scandir(log_dir) if f.is_dir() ]
    # with Pool(20) as p:
    # #     p.map(overlay_simulation_traces, perfectly_unsafe_scenarios)
    # log_dir = 'results/perfectly_unsafe'
    # fig = overlay_simulation_traces((log_dir, 'scenario_080000000071e0c19'))
    # fig.show()
    # trace 5 is bad
    # print(scenario.settings)
    # scenario.settings[0]['basal_iq'] = True
    # scenario.user_config = UserConfig(resume=True)
    # log_dir = 'results/perfectly_unsafe'
    # scenario, verification_traces, safety= load_from_dir(log_dir, 'scenario_08000000006d3ff3c')
    # init = get_init(scenario, 1)
    # print(init)
    # traces = simulate_from_init(scenario, init, logging=True, log_dir='results/logs')
    # plot_variable(traces, 'G')
    # results = load_results('results/verification')
    # print(unsafe_analysis(results, 2))

    verify_wrapper()
