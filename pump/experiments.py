from artificial_pancreas_simulate import *
from verse_model import *
from artificial_pancreas_agent import *
from pump_model import *
from cgm import *
from hovorka_model import HovorkaModel
import pickle
import random


def generate_scenarios(config, num_scenarios):
    # first we will build the meal config
    meals_low: List[Meal] = []
    meals_high: List[Meal] = []
    
    boluses: List[Bolus] = []
    
    num_meals = random.randint(config['meals']['num_meals']['low'], config['meals']['num_meals']['high'])
    intervals = random.randint(config['meals']['interval']['low'], config['meals']['interval']['high'], (num_meals - 1,))
    intervals = np.insert(intervals, 0, 0, axis=0)
    meal_times = np.cumsum(intervals)
    
    bolus = np.random.binomial(config['meals'][''])
    
    for i in range(num_meals):
        meal_range = random.choice(config['meals']['intervals'])
        meal_low = Meal(meal_times[i], meal_range['low'])
        meal_high = Meal(meal_times[i], meal_range['high'])
        meals_low.append(meal_low)
        meals_high.append(meal_high)
        
        # should we bolus for this meal, and if so, when
        bolus = np.random.binomial()
    
    


def iob_correction_demo(settings):
    BW = 70  # kg
    basal = 0  # units
    boluses = []
    boluses = []
    meals = [Meal(0, 100), Meal(200, 100)]
    boluses = [Bolus(10, 100, BolusType.Simple, None), Bolus(200, 100, BolusType.Simple, None)]
    num_meals = len(meals)
    
    traces = simulate_multi_meal_scenario(100, BW, basal, boluses, meals, duration=(num_meals + 10) * 60, settings=settings)
    linear_transform_trace(traces, 'pump', state_indices['iob'] + 1, 0.12 * 70, 0) # + 1 because time is index 0
    fig1 = plot_variable(traces, 'iob')
    fig5 = plot_variable(traces, 'G')
    
    
if __name__ == '__main__':
    settings = {
        'carb_ratio': 25,
        'correction_factor': 60,
        'insulin_duration': 420,
        'max_bolus': 15,
        'basal_rate': 0.366,
        'target_bg': 130
    }
    iob_correction_demo(settings)
    