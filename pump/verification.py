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
    meal_carb_ranges = [(0, 50), (50, 100), (100, 150)]
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
        if low_sum >= TOTAL_LOW and high_sum <= TOTAL_HIGH:
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
    
    ERROR_LOW = 0.9
    ERROR_HIGH = 1.1
        
    PATIENT_BASAL_GLUCOSE = 6.5
    CGM_BIAS = 0.1
    NUM_MEALS = 4
    
    BOLUS_OFFSET = -5
    
    BASAL_RATE_RANGE = 0.1
    
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
        scenario = SimulationScenario(init_bg, boluses, meals, errors, [settings_low, settings_high], patient_params, cgm_config)
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

def run_verification_scenario(scenario):
    tqdm.write(str(scenario))
    traces = verify_multi_meal_scenario(scenario)
    safety_results = evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))
    save_scenario_results(scenario, traces, safety_results, 'results/verification')

def sigint(signum, frame):
    os.kill(0, signal.SIGKILL)

def verify(scenarios: List[SimulationScenario], pool_size: int):
    with Pool(pool_size) as p:
        p.map(run_verification_scenario, scenarios)
    
if __name__ == '__main__':
    # random.seed(42)
    # np.random.seed(42)
    # scenarios = []
    # g = gen_verification_scenarios()
    # for i in range(2):
    #     s = next(g)
    #     s.sim_duration = 60
    #     scenarios.append(s)
    # verify(scenarios, pool_size=2)    

    parser = argparse.ArgumentParser('pumpverif')
    parser.add_argument('-p', '--processes', default=1, type=int)
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, sigint)
    np.random.seed(42)
    scenarios = gen_verification_scenarios()
    np.random.shuffle(scenarios)  
    print(args.processes)
    verify(scenarios, pool_size=args.processes)
        