from artificial_pancreas_simulate import *
from verse_model import *
from artificial_pancreas_agent import *
from pump_model import *
from cgm import *
from hovorka_model import HovorkaModel
import pickle
import random
import json


def generate_scenario(config):
    # first we will build the meal config
    meals_low: List[Meal] = []
    meals_high: List[Meal] = []
    
    boluses: List[Bolus] = []
    
    num_meals = np.random.randint(config['meals']['number']['low'], config['meals']['number']['high'])
    intervals = np.random.randint(config['meals']['interval']['low'], config['meals']['interval']['high'], size=(num_meals - 1,))
    intervals = np.insert(intervals, 0, 0, axis=0)
    meal_times = np.cumsum(intervals)
    
    forget_prob = config['meals']['forget_prob']
    forget_delay = config['meals']['forget_delay']
    bolus = np.random.binomial(size=num_meals, n=1, p=forget_prob)
    
    for i in range(num_meals):
        meal_range = random.choice(config['meals']['carbs'])
        meal_low = Meal(meal_times[i], meal_range['low'])
        meal_high = Meal(meal_times[i], meal_range['high'])
        meals_low.append(meal_low)
        meals_high.append(meal_high)
        
        # should we bolus for this meal, and if so, when
        if bolus[i]:
            boluses.append(Bolus(meal_times[i], -1, BolusType.Simple, None))
        else:
            # if the user forgot, do they realize later?
            delay = np.random.randint(forget_delay['low'], forget_delay['high'])
            boluses.append(Bolus(meal_times[i] + delay, -1, BolusType.Simple, None)) # TODO: handle correction here
    
    
    init_bg_range = random.choice(config['patient']['init_bg'])
    init_bg = [init_bg_range['low'], init_bg_range['high']]
    
    settings = get_recommended_settings()
    
    patient_params = random.choice(config['patient']['parameters'])
    
    # we need some buffer for the duration
    duration = meal_times[-1] + config['misc']['duration_buffer']
    
    return {
        'meals': [meals_low, meals_high],
        'boluses': boluses,
        'init_bg': init_bg,
        'settings': settings,
        'patient': patient_params,
        'duration': duration
    }
    
def run_scenario(scenario):
    traces = verify_multi_meal_scenario(scenario['init_bg'], scenario['patient']['BW'], None, scenario['boluses'], scenario['meals'], scenario['duration'], scenario['settings'])
    return traces        
        
    
    


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
    with open('./pump/configurations/testing_config.json') as f:
        config = json.load(f)
    scenario = generate_scenario(config)
    traces = run_scenario(scenario)
    breakpoint()