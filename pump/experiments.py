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
    
    def analyze(self, traces, scenario):
        glucose_trace = extract_variable(traces, 'pump', state_indices['G'] + 1)
        safety_analysis = tir_analysis(glucose_trace, self.config['safety']['tir']['low'], self.config['safety']['tir']['high'])
        realism_analysis = realism(scenario, config['safety']['realism']['carb_to_weight_low'], config['safety']['realism']['carb_to_weight_high'])
        return {'safety': safety_analysis, 'realism': realism_analysis}

def generate_scenario(config):
    # PATIENT PARAMETERS
    GBasal = np.random.choice(config['patient']['parameters'])['GBasal']
    patient_params = patient_original({'basalGlucose': GBasal})
    TauM_DEFAULT = patient_params['TauM']
    EXTENDED_END_BUFFER = 5
    
    # PUMP SETTINGS
    # FIXME: there is probably a better way to handle this
    settings = get_recommended_settings(BW=patient_params['w'], TDD=patient_params['TDD'])
    basal_iq = np.random.choice(config['settings']['basal_iq'])
    settings['basal_iq'] = basal_iq
    
    basal_rate_multiplier = np.random.choice(config['settings']['basal_rate'])
    b_low = basal_rate_multiplier["low"]
    b_high = basal_rate_multiplier["high"]
    settings_low = settings.copy()
    settings_high = settings.copy()
    settings_low['basal_rate'] = settings['basal_rate'] * b_low
    settings_high['basal_rate'] = settings['basal_rate'] * b_high
    
    # INITIAL CONDITIONS
    init_bg_range = np.random.choice(config['patient']['init_bg'])
    init_bg = [init_bg_range['low'], init_bg_range['high']]    
    
    # MEALS AND BOLUSES
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
    
    extended_end = -1 # when will the current extended bolus end? (we can't have simulateous extended boluses)
    
    for i in range(num_meals):
        meal_range = np.random.choice(config['meals']['carbs'])
        
        if meal_times[i] + EXTENDED_END_BUFFER <= extended_end:
            TauM = TauM_DEFAULT
        else:
            TauM = np.random.choice(config['meals']['TauM'][0], p=config['meals']['TauM'][1])
        
        meal_low = Meal(meal_times[i], meal_range['low'], TauM)
        meal_high = Meal(meal_times[i], meal_range['high'], TauM)
        meals_low.append(meal_low)
        meals_high.append(meal_high)
        
        # should we bolus for this meal, and if so, when
        
        # TODO: this should be moved into a "user-agent"
        
        bolus_type = BolusType.Simple
        bolus_config = None
        if TauM != TauM_DEFAULT:
            extended_config = np.random.choice(config['boluses']['extended'])
            bolus_type = BolusType.Extended
            bolus_config = ExtendedBolusConfig(extended_config['percentage'], extended_config['duration'])
        
        if bolus[i]:
            boluses.append(Bolus(meal_times[i], -1, bolus_type, bolus_config))
        else:
            # if the user forgot, do they realize later?
            delay = np.random.randint(forget_delay['low'], forget_delay['high'])
            boluses.append(Bolus(meal_times[i] + delay, FORGOT_BOLUS, bolus_type, bolus_config)) # TODO: handle correction here

        if bolus_type == BolusType.Extended:
            extended_end = meal_times[i] + bolus_config.duration
    
    # SIMULATION DURATION
    duration = meal_times[-1] + config['misc']['duration_buffer']
    
    return freeze({
        'meals': [meals_low, meals_high],
        'boluses': boluses,
        'init_bg': init_bg,
        'settings': [settings_low, settings_high],
        'patient': patient_params,
        'duration': duration
    })
    
def check_scenario(scenario: SimulationScenario):
    # source: https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/carbohydrates/art-20045705
    TOTAL_CARBS_LOW = 100
    TOTAL_CARBS_HIGH = 350
    MEAL_TIME_RANGES = [(60, 360), (360, 600), (600, 840), (840, 1080)]
    INTER_MEAL_TIME = 30
    ALLOWED_TMAX = [DEFAULT_MEAL, HIGH_FAT_MEAL]
    TMAX_TO_CONFIG = {DEFAULT_MEAL: [(BolusType.Simple, None)], HIGH_FAT_MEAL: [(BolusType.Extended, ExtendedBolusConfig(50, 120))]}
    BOLUS_MEAL_DELTA = 20
    
    meals = scenario.get_meals()
    
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
    if not len(meals) == 4:
        print('bad number of meals')
        return False
    # we will start scenario at 5 AM
    for i in len(MEAL_TIME_RANGES):
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
            
    
def verify():
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
    while True:
        meal_1_time = np.random.choice(60 * np.array([1, 2, 3, 4, 5, 6]))
        meal_2_time = np.random.choice(60 * np.array([5, 6, 7, 8, 9, 10]))
        meal_3_time = np.random.choice(60 * np.array([10, 11, 12, 13, 14]))
        meal_4_time = np.random.choice(60 * np.array([14, 15, 16, 17, 18]))
        
        meal_TauMs = np.random.choice([DEFAULT_MEAL, HIGH_FAT_MEAL], (4,))
        
        bolus_configs = {DEFAULT_MEAL: (BolusType.Simple, None), HIGH_FAT_MEAL: (BolusType.Extended, ExtendedBolusConfig(50, 3))}
        
        meal_carb_ranges = [(0, 50), (50, 100), (100, 150)]
        meal_carbs = [random.choice(meal_carb_ranges) for _ in range(4)]
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
        
        scenario = SimulationScenario(init_bg, boluses, meals, errors, [settings, settings], patient_params)
        
        if not check_scenario(scenario):
            continue
                
        verify_multi_meal_scenario(scenario)
    pass
    
def run_scenario(scenario, log_dir):
    
    # HACK    
    # Notes
    # we can do scenario['settings'][0]['basal_iq'] since we don't vary basal IQ in pump settings
    traces = verify_multi_meal_scenario(scenario['init_bg'], thaw(scenario['patient']), scenario['settings'][0]['basal_iq'], scenario['boluses'], scenario['meals'], scenario['duration'], scenario['settings'], log_dir=log_dir)
    return traces   

def test(config, num_scenarios, safety_analyzer: SafetyAnalyzer, log_dir):
    state_file = os.path.join(log_dir, f'random_state.pickle')

    def save_random_state():
        np_state = np.random.get_state()
        py_state = random.getstate()
        random_state = (np_state, py_state)
        with open(state_file, 'wb') as f:
            pickle.dump(random_state, f)

    def load_random_state():
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                (np_state, py_state) = pickle.load(f)
            np.random.set_state(np_state)
            random.setstate(py_state)
        else:
            np.random.seed(config['misc']['random_seed'])
            random.seed(config['misc']['random_seed'])
    
    scenarios_tested = set()
    
    # load already executed scenarios
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() ]
    scenario_idx = 0
    for dir in scenario_dirs:
        try:
            with open(os.path.join(dir.path, 'scenario.pkl'), 'rb') as f:
                scenario = pickle.load(f)
            scenarios_tested.add(scenario)
            scenario_idx = max(scenario_idx, int(dir.name[9:])) # FIXME: there is probably a better way to take out the scenario_ prefix
        except:
            # this scenario was not fully tested, so we need to re-test
            rmtree(dir)
    
    scenario_idx += 1
    
    # before we start fuzzing, load state
    load_random_state()
    
    while len(scenarios_tested) < num_scenarios:
        
        # before generating any scenarios, dump random state
        save_random_state()
        while (scenario := generate_scenario(config)) in scenarios_tested:
            pass                
        scenario_log_dir = os.path.join(log_dir, f'scenario_{scenario_idx}')
        os.makedirs(scenario_log_dir)
        traces = run_scenario(scenario, scenario_log_dir)
        with open(os.path.join(scenario_log_dir, 'traces.pkl'), 'wb+') as f:
            pickle.dump(traces, f)
        with open(os.path.join(scenario_log_dir, 'scenario.pkl'), 'wb+') as f:
            pickle.dump(scenario, f)
        with open(os.path.join(scenario_log_dir, 'scenario.json'), 'w+') as f:
            json.dump(thaw(scenario), f, cls=ScenarioEncoder, indent=4)
        
         
        analysis_results = safety_analyzer.analyze(traces, scenario)  
        with open(os.path.join(scenario_log_dir, 'safety.json'), 'w+') as f:
            json.dump(analysis_results, f)
        scenario_idx += 1
        scenarios_tested.add(scenario)
    

def plot_scenario(log_dir, index, variable, show=False, problem_dirs=[]):
    
    if os.path.join(log_dir, f'scenario_{index}') in problem_dirs:
        print('bad scenario')
        return 
    
    with open(os.path.join(log_dir, f'scenario_{index}', 'traces.pkl'), 'rb') as f:
        traces = pickle.load(f)
    fig = plot_variable(traces, variable, show=show)
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
    

def basaliq_empirical_test(num_samples):
    settings = get_recommended_settings(TDD=39.22)
    BW = 74.9  # kg
    basal = 0  # units
    params = patient_original({'basalGlucose': 6.5})
    settings['basal_rate'] = 1.5
    settings['carb_ratio'] = 10
    settings['max_bolus'] = 20
    meals_low = [Meal(0, 30), Meal(240, 60), Meal(600, 50), Meal(840, 20)]
    meals_high = [Meal(0, 50), Meal(240, 100), Meal(600, 70), Meal(840, 40)]
    boluses = [Bolus(0, -1, BolusType.Simple, None), Bolus(240, -1, BolusType.Simple, None), Bolus(600, -1, BolusType.Simple, None), Bolus(840, -1, BolusType.Simple, None)]
    if not os.path.exists('results/basaliq/traces.pkl'):
        traces = verify_multi_meal_scenario([80, 120], params, True, boluses, [meals_low, meals_high], duration=24 * 60, settings=settings, logging=False)
        with open('results/basaliq/traces.pkl', 'wb+') as f:
            pickle.dump(traces, f)
    
    # now randomly sample from distribution and check containment
    for i in range(num_samples):
        # get a sample
        bg = np.random.uniform(80, 120)
        
        m1 = np.random.uniform(30, 50)
        m2 = np.random.uniform(60, 100)
        m3 = np.random.uniform(50, 70)
        m4 = np.random.uniform(20, 40)
        
        print(f'simulating({bg}, {m1}, {m2}, {m3}, {m4})')
        
        meals = [Meal(0, m1), Meal(240, m2), Meal(600, m3), Meal(840, m4)]
    
    
def fuzz(config_file, num_scenarios, results_path):
    with open(config_file) as f:
        config = json.load(f)
    seed = config['misc']['random_seed']
    np.random.seed(seed)
    random.seed(seed)
    safety_analyzer = SafetyAnalyzer(config['safety'])
    test(config, num_scenarios, safety_analyzer, results_path)
    
    
def rank_scenarios(log_dir):
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() ]
    keys = {}
    for dir in scenario_dirs:
        idx = int(dir.name[9:])
        with open(os.path.join(dir.path, 'safety.json'), 'rb') as f:
            safety = json.load(f)
            metric = safety['tir']['low']
            keys[idx] = metric
    sorted_keys = list(sorted(keys.keys(), key=keys.get))
    return sorted_keys    


# Test different meal amounts, starting BGs, and basal rates (around the ideal)
# Which combination of deliver now percentage and duration gives the best results?
def find_optimal_extended_settings():
    settings = get_recommended_settings(TDD=39.22, BW=74.9)
    BW = 74.9  # kg
    basal = 0  # units
    params = patient_original({'basalGlucose': 6.5})
    settings['basal_rate'] = params['Ub'] # set ideal basal rate
    deliver_now = [20, 30, 40, 50, 60, 70]
    duration = [1, 2, 3, 4, 5, 6]
    
    meals = [Meal(60, 60, 120)]
    # simple bolus
    boluses = [Bolus(55, None, BolusType.Simple, 0, True, None)]
    traces = simulate_multi_meal_scenario(120, params, False, boluses, meals, duration=15 * 60, settings=settings, logging=False)
    tir = tir_analysis_simulate(extract_variable(traces, 'pump', state_indices['G'] + 1, True))
    variation = tir['high'] - tir['low']
    print('simple', tir, variation)
    for dn in deliver_now:
        for dur in duration:
            boluses = [Bolus(55, None, BolusType.Extended, 0, True, ExtendedBolusConfig(dn, dur * 60))]
            traces = simulate_multi_meal_scenario(120, params, False, boluses, meals, duration=15 * 60, settings=settings, logging=False)
            tir = tir_analysis_simulate(extract_variable(traces, 'pump', state_indices['G'] + 1, True))
            variation = tir['high'] - tir['low']
            print(dn, dur, tir, variation)
    

def fixup_safety(log_dir):
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() ]
    for dir in scenario_dirs:
        try:
            with open(os.path.join(dir.path, 'safety.json'), 'rb') as f:
                safety = json.load(f)
            r = safety['realism']
            f_low = 3
            f_high = 6
            BW = 74.9
            cl = r['carbs_low']
            ch = r['carbs_high']
            dl = r['low_distance']
            dh = r['high_distance']
            if dh != max(0, ch - f_high * BW):
                r['high_distance'] = max(0, ch - f_high * BW)
                print('dumping')
                with open(os.path.join(dir.path, 'safety.json'), 'w') as f:
                    json.dump(safety, f)
        except:
            pass  

# problem_dirs contains dirs that were affected by the extended bolus emulation bug
def realism_safety_plot(log_dir, realism_func, safety_func, problem_dirs=[]):
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() if f.path not in problem_dirs]
    
    realism_vals = []
    safety_vals = []
    biqs = []
    indices = []
    
    for dir in scenario_dirs:
        try:
            with open(os.path.join(dir.path, 'safety.json'), 'rb') as f:
                safety = json.load(f)
            with open(os.path.join(dir.path, 'scenario.json'), 'r') as f:
                scenario = json.load(f)
            biq = scenario['settings'][0]['basal_iq']
            biqs.append(biq)
            realism_vals.append(realism_func(safety))
            safety_vals.append(safety_func(safety))
            scenario_idx = int(dir.name[9:])
            indices.append(scenario_idx)
        except:
            pass
    fig, ax = plt.subplots()
    c = ['g' if biqs[i] else 'r' for i in range(len(biqs))]
    
    safety_biq = [safety_vals[i] for i in range(len(biqs)) if biqs[i]]
    realism_biq = [realism_vals[i] for i in range(len(biqs)) if biqs[i]]
    
    safety_nobiq = [safety_vals[i] for i in range(len(biqs)) if not biqs[i]]
    realism_nobiq = [realism_vals[i] for i in range(len(biqs)) if not biqs[i]]
    
    biq = ax.scatter(realism_biq, safety_biq, c='g')
    nobiq = ax.scatter(realism_nobiq, safety_nobiq, c='r')
    
    plt.legend((biq, nobiq), ('Basal-IQ enabled', 'Basal-IQ Disabled'))
    
    for i in range(len(realism_vals)):        
        ax.annotate(f'{indices[i]}', (realism_vals[i], safety_vals[i]))
    plt.xlabel('Realism Metric')
    plt.ylabel('Safety Metric')
    plt.title('Safety vs Realism')
    plt.show()  
    
def find_bad_runs(log_dir):
    scenario_dirs = [ f for f in os.scandir(log_dir) if f.is_dir() ]
    rerun_dirs = []
    count = 0
    for dir in scenario_dirs:
        with open(os.path.join(dir.path, 'scenario.json'), 'r') as f:
            scenario = json.load(f)
        boluses = scenario['boluses']
        extended_bolus_times = []
        for bolus in boluses:
            if bolus['type'] == 'Extended':
                extended_bolus_times.append(bolus['time'])
              
        # check if the bolus ever failed to deliver
        fail = False
        for sim in range(11):
            with open(os.path.join(dir.path, f'sim_{sim}_dose.txt')) as f:
                doses = f.read().splitlines()        
            for t in extended_bolus_times:
                dose_t = float(doses[t].split("=")[1].strip())
                if dose_t < 1:
                    fail = True
                break
            if fail:
                break
        if fail:
            rerun_dirs.append(dir.path)
    return rerun_dirs
            
if __name__ == '__main__':
    # with open('pump/configurations/testing_config.json', 'r') as f:
    #     config = json.load(f)
    # safety_analyzer = SafetyAnalyzer(config)
    # test(config, 200, safety_analyzer, 'results/remote/fuzzing')
    # fig = plot_scenario('results/remote/fuzzing', 6, 'G')
    # fig.write_image('scenario.png')

    # realism_func = lambda safety: safety['realism']['carbs_high']
    # safety_func = lambda safety: safety['safety']['tir']['high']
    
    # problem_dirs = find_bad_runs('results/remote/fuzzing')
    # realism_safety_plot('results/remote/fuzzing', realism_func, safety_func, problem_dirs)
    # plot_scenario('results/remote/fuzzing', 90, 'G', True, problem_dirs)
    
    # show how long they were in different ranges
    
    # 300 for 3 hours vs 600 for 1 hour
    
    # 600: go to ER
    
    verify()