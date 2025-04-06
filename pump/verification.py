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
import argparse
from shutil import rmtree
from simutils import FORGOT_BOLUS
from safety.safety import realism
import matplotlib.pyplot as plt
from multiprocessing import Pool
import threading
import os
import signal

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

def get_allowed_meal_carb_ranges(TOTAL_LOW, TOTAL_HIGH):
    meal_carb_ranges = [(0, 37.5), (37.5, 75), (75, 112.5), (112.5, 150)]
    good_ranges = []
    for i in range(64):
        n = i
        idx0 = n % 4
        n  = n // 4
        idx1 = n % 4
        n = n // 4
        idx2 = n % 4
        n = n // 4
        idx3 = n % 4
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
    
    meal_ranges = get_allowed_meal_carb_ranges(100, 350)
    
    while True:
        meal_1_time = np.random.choice(60 * np.array([1, 2, 3, 4, 5]))
        meal_2_time = np.random.choice(60 * np.array([6, 7, 8, 9, 10]))
        meal_3_time = np.random.choice(60 * np.array([11, 12, 13, 14]))
        meal_4_time = np.random.choice(60 * np.array([15, 16, 17, 18]))
        
        meal_TauMs = np.random.choice([DEFAULT_MEAL, HIGH_FAT_MEAL], (4,))
        
        bolus_configs = {DEFAULT_MEAL: (BolusType.Simple, None), HIGH_FAT_MEAL: (BolusType.Extended, ExtendedBolusConfig(50, 180))}
        
        meal_carbs = random.choice(meal_ranges)
        meals = [Meal(meal_1_time, meal_carbs[0], meal_TauMs[0]), Meal(meal_2_time, meal_carbs[1], meal_TauMs[1]), Meal(meal_3_time, meal_carbs[2], meal_TauMs[2]), Meal(meal_4_time, meal_carbs[3], meal_TauMs[3])]
        
        bolus_offsets = np.random.choice([-5], (4,))
        
        boluses = []
        for i in range(len(meals)):
            bolus = Bolus(meals[i].time + bolus_offsets[i], None, None, i, True, None)
            bolus = set_bolus_config(bolus, bolus_configs[meals[i].TauM])
            boluses.append(bolus)
            
        errors = [ERROR_LOW, ERROR_HIGH]
        
        init_bg = [70, 180]
        
        patient_params = patient_original({'basalGlucose': PATIENT_BASAL_GLUCOSE})
        settings = get_recommended_settings(TDD=39.22, BW=74.9)
        settings['basal_rate'] = patient_params['Ub']
        settings['basal_iq'] = True
        
        cgm_config = CGMConfig((1 - CGM_BIAS, 1 + CGM_BIAS), (0, 0))
        
        scenario = SimulationScenario(init_bg, boluses, meals, errors, [settings, settings], patient_params, cgm_config)
        
        if not check_scenario(scenario):
            continue
        
        yield scenario

def run_verification_scenario(scenario):
    traces = verify_multi_meal_scenario(scenario)
    safety_results = evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))
    print(scenario)
    print(safety_results)

def sigint(signum, frame):
    os.kill(0, signal.SIGKILL)

def verify(scenarios: List[SimulationScenario], pool_size: int):
    with Pool(pool_size) as p:
        p.map(run_verification_scenario, scenarios)
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint)
    scenarios = []
    g = gen_verification_scenarios()
    for i in range(5):
        s = next(g)
        s.sim_duration = 1
        scenarios.append(s)
    print(scenarios)
    verify(scenarios, pool_size=5)    
    
        