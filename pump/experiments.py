from artificial_pancreas_simulate import *
from verse_model import *
from artificial_pancreas_agent import *
from pump_model import *
from cgm import *
from hovorka_model import HovorkaModel, patient_original
import pickle
import random
import json
from pyrsistent import freeze, thaw
from dataclasses import asdict

# this is currently only to provide a human-readable representation of a scenario
# the actual serialization/deserialization is done by pickling
class ScenarioEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
    

class SafetyAnalyzer:
    def __init__(self, config):
        self.config = config
    
    def analyze(self, traces):
        glucose_trace = extract_variable(traces, 'pump', state_indices['G'] + 1)
        tir_low, tir_high = tir_analysis(glucose_trace, self.config['tir']['low'], self.config['tir']['high'])
        return {
            'tir': {
                'lb': tir_low,
                'ub': tir_high
            }
        }

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
    bolus = np.random.binomial(size=num_meals, n=1, p= 1 - forget_prob) # probability that bolus happens at meal time is 1 - forget_prob
    
    for i in range(num_meals):
        meal_range = np.random.choice(config['meals']['carbs'])
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
            boluses.append(Bolus(meal_times[i] + delay, 0, BolusType.Simple, None)) # TODO: handle correction here
    
    
    init_bg_range = np.random.choice(config['patient']['init_bg'])
    init_bg = [init_bg_range['low'], init_bg_range['high']]
    
    GBasal = np.random.choice(config['patient']['parameters'])['GBasal']
    patient_params = patient_original({'basalGlucose': GBasal})
    
    settings = get_recommended_settings(BW=patient_params['w'], TDD=patient_params['TDD'])
    
    # FIXME: there is probably a better way to handle this
    basal_iq = np.random.choice(config['settings']['basal_iq'])
    settings['basal_iq'] = basal_iq
    
    
    # we need some buffer for the duration
    duration = meal_times[-1] + config['misc']['duration_buffer']
    
    return freeze({
        'meals': [meals_low, meals_high],
        'boluses': boluses,
        'init_bg': init_bg,
        'settings': settings,
        'patient': patient_params,
        'duration': duration
    })
    
def run_scenario(scenario, log_dir):
    
    # HACK    
    traces = verify_multi_meal_scenario(scenario['init_bg'], thaw(scenario['patient']), scenario['settings']['basal_iq'], scenario['boluses'], scenario['meals'], scenario['duration'], scenario['settings'], log_dir=log_dir)
    return traces   

def test(config, num_scenarios, safety_analyzer: SafetyAnalyzer, log_dir):
    scenarios_tested = set()
    
    # load already executed scenarios
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() ]
    scenario_idx = 0
    for dir in scenario_dirs:
        with open(os.path.join(dir.path, 'scenario.pkl'), 'rb') as f:
            scenario = pickle.load(f)
        scenarios_tested.add(scenario)
        scenario_idx = max(scenario_idx, int(dir.name[9:])) # FIXME: there is probably a better way to take out the scenario_ prefix
    
    scenario_idx += 1
    
    while len(scenarios_tested) < num_scenarios:
        while (scenario := generate_scenario(config)) in scenarios_tested:
            pass
        scenario_log_dir = os.path.join(log_dir, f'scenario_{scenario_idx}')
        os.makedirs(scenario_log_dir)
        traces = run_scenario(scenario, scenario_log_dir)
        plot_variable(traces, 'G')
        analysis_results = safety_analyzer.analyze(traces)  
        with open(os.path.join(scenario_log_dir, 'safety.json'), 'w+') as f:
            json.dump(analysis_results, f)
        with open(os.path.join(scenario_log_dir, 'traces.pkl'), 'wb+') as f:
            pickle.dump(traces, f)
        with open(os.path.join(scenario_log_dir, 'scenario.pkl'), 'wb+') as f:
            pickle.dump(scenario, f)
        with open(os.path.join(scenario_log_dir, 'scenario.json'), 'w+') as f:
            json.dump(thaw(scenario), f, cls=ScenarioEncoder, indent=4)
        scenario_idx += 1
        scenarios_tested.add(scenario)
    

def plot_scenario(log_dir, index, variable):
    with open(os.path.join(log_dir, f'scenario_{index}', 'traces.pkl'), 'rb') as f:
        traces = pickle.load(f)
    fig = plot_variable(traces, variable)
    return fig
    
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
    seed = config['misc']['random_seed']
    np.random.seed(seed)
    random.seed(seed)
    safety_analyzer = SafetyAnalyzer(config['safety'])
    scenario = generate_scenario(config)
    test(config, 20, safety_analyzer, 'results/fuzzing_translate')
