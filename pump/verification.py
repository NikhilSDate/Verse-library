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
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from multiprocessing import Pool
import os
import signal
import itertools
import ast
from tqdm import tqdm
from typing import Any
from functools import partial
import time
import gzip


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
    meal_carb_ranges = [(0, 40), (40, 80), (80, 120), (120, 160)]
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
    
def gen_verification_scenarios() -> List[SimulationScenario]:
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
    print(len(meal_ranges))
    
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
    result = ''
    while True:
        attempt = os.path.join(output_dir, f'scenario_{idx}')
        if not os.path.exists(attempt):
            result = attempt
            break
        idx += 1
    os.makedirs(result)
    return result

def get_log_directory(scenario: SimulationScenario, output_dir):
    idx = 0
    result = ''
    while True:
        attempt = os.path.join(output_dir, 'logs', f'log_{idx}')
        if not os.path.exists(attempt):
            result = attempt
            break
        idx += 1
    os.makedirs(result)
    return result

def save_result_with_sims(result: Tuple[SimulationScenario, object, object], output_dir):
    scenario, traces, safety = result
    scenario_directory = get_scenario_directory(scenario, output_dir)
    if scenario_directory is None:
        print('redundant scenario')
        return
    fig = plot_variable(traces, 'G', show=False)
    fig.write_image(os.path.join(scenario_directory, 'plot.png'))
    with gzip.open(os.path.join(scenario_directory, 'traces.gzip'), 'wb') as f:
        pickle.dump(traces, f)
    with open(os.path.join(scenario_directory, 'safety.txt'), 'w') as f:
        f.write(str(safety))
    with open(os.path.join(scenario_directory, 'scenario.yaml'), 'w') as f:
        to_dump = denumpify(asdict(scenario.get_data(), dict_factory=custom_asdict_factory))
        yaml.dump(to_dump, f)
    with open(os.path.join(scenario_directory, 'scenario.pkl'), 'wb') as f:
        pickle.dump(scenario, f)
    fig_with_sims = plot_results(result)
    fig_with_sims.write_image(os.path.join(scenario_directory, 'plot_with_sims.png'))

def save_scenario_results(scenario: SimulationScenario, traces, safety_results, output_dir):
    # create a directory in output_dir using hash of scenario
    scenario_directory = get_scenario_directory(scenario, output_dir)
    if scenario_directory is None:
        print('redundant scenario')
        return
    fig = plot_results((scenario, traces, safety_results))
    with gzip.open(os.path.join(scenario_directory, 'traces.gzip'), 'wb') as f:
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

def save_scenario_runtime(scenario, output_dir, runtime):
    log_dir = get_log_directory(scenario, output_dir)
    with open(os.path.join(log_dir, 'runtime.txt'), 'w') as f:
        f.write(str(runtime))

def run_verification_scenario(scenario, output_dir):
    start_time = time.time()
    res = verify_multi_meal_scenario(scenario)
    if res.type == ResultType.OK:
        traces = res.payload
        safety_results = evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))
        save_scenario_results(scenario, traces, safety_results, output_dir)
    else:
        save_crash(scenario, res.payload, output_dir)
    end_time = time.time()
    runtime = end_time - start_time
    save_scenario_runtime(scenario, output_dir, runtime)

def sigint(signum, frame):
    os.kill(0, signal.SIGKILL)

def verify(scenarios: List[SimulationScenario], output_dir: str, pool_size: int):
    run_func = partial(run_verification_scenario, output_dir=output_dir)
    with Pool(pool_size) as p:
        p.map(run_func, scenarios)

# TODO: consider rewriting this with generators
def load_scenarios(output_dir) -> List[SimulationScenario]:
    scenarios = []
    scenario_dirs = [ f for f in os.scandir(output_dir) if f.is_dir() ] if os.path.exists(output_dir) else []
    for scenario_dir in tqdm(scenario_dirs):
        try:
            with open(os.path.join(scenario_dir.path, 'scenario.pkl'), 'rb') as f:
                scenario = pickle.load(f)
            scenarios.append(scenario)
        except:
            pass
    return scenarios

# returns a mapping of scenarios to paths
# (allows easy lookup of traces from disk in a secondary pass)
def load_scenarios_and_dirs(result_dir: str) -> Dict[SimulationScenario, Tuple[str, str]]:
    scenarios = {}
    scenario_dirs = [ f for f in os.scandir(result_dir) if f.is_dir() ] if os.path.exists(result_dir) else []
    for scenario_dir in tqdm(scenario_dirs):
        try:
            with open(os.path.join(scenario_dir.path, 'scenario.pkl'), 'rb') as f:
                scenario = pickle.load(f)
            scenarios[scenario] = (result_dir, scenario_dir.path)
        except:
            pass
    return scenarios


def load_results(output_dir) -> List[Tuple[SimulationScenario, object, object]]:
    results = []
    scenario_dirs = [ f for f in os.scandir(output_dir) if f.is_dir() ]
    for scenario_dir in tqdm(scenario_dirs):
        try:
            with open(os.path.join(scenario_dir.path, 'scenario.pkl'), 'rb') as f:
                scenario = pickle.load(f)
            with gzip.open(os.path.join(scenario_dir.path, 'traces.gzip'), 'rb') as f:
                traces = pickle.load(f)
            with open(os.path.join(scenario_dir.path, 'safety.txt')) as f:
                safety = ast.literal_eval(f.read())
            results.append((scenario, traces, safety))
        except FileNotFoundError:
            pass
    return results  

def load_results_gen(output_dir) -> Generator[Tuple[SimulationScenario, AnalysisTree, object], None, None]:
    for scenario_dir in os.scandir(output_dir):
        if not scenario_dir.is_dir():
            continue
        try:
            with open(os.path.join(scenario_dir.path, 'scenario.pkl'), 'rb') as f:
                scenario = pickle.load(f)
            with gzip.open(os.path.join(scenario_dir.path, 'traces.gzip'), 'rb') as f:
                traces = pickle.load(f)
            with open(os.path.join(scenario_dir.path, 'safety.txt')) as f:
                safety = ast.literal_eval(f.read())
            yield (scenario, traces, safety)
        except:
            continue

def load_from_dir(output_dir, scenario_dir) -> Tuple[SimulationScenario, AnalysisTree, List[bool]]:
    scenario_dir = os.path.join(output_dir, scenario_dir)
    scenario, traces, safety = None, None, None
    scenario_path = os.path.join(scenario_dir, 'scenario.pkl')
    traces_path = os.path.join(scenario_dir, 'traces.gzip')
    safety_path = os.path.join(scenario_dir, 'safety.txt')
    if os.path.exists(scenario_path):
        with open(scenario_path, 'rb') as f:
            scenario = pickle.load(f)
    if os.path.exists(traces_path):
        with gzip.open(traces_path, 'rb') as f:
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

def debug_sim(output_dir, result_dir, sim_idx):
    scenario, traces, safety = load_from_dir(output_dir, result_dir)
    init = get_init(traces, sim_idx)
    traces = simulate_from_init(scenario, init, logging=True, log_dir=os.path.join(output_dir, result_dir, 'debug', f'sim_{sim_idx}'))
    fig = plot_variable(traces, 'G', show=False)
    fig.write_image(os.path.join(output_dir, result_dir, 'debug', f'sim_{sim_idx}', 'plot.png'))

def get_scenarios_to_run(output_dir, node_count, node_idx):
    scenarios = gen_verification_scenarios()
    np.random.shuffle(scenarios)
    scenarios = [scenario for i, scenario in enumerate(scenarios) if i % node_count == node_idx]
    existing = set(load_scenarios(output_dir))
    scenarios = [scenario for scenario in scenarios if scenario not in existing]
    return scenarios

def verify_wrapper():
    parser = argparse.ArgumentParser('pumpverif')
    parser.add_argument('-p', '--processes', default=1, type=int)
    parser.add_argument('-s', '--seed', default=42, type=int)
    parser.add_argument('-o', '--output-dir', default='results/verification', type=str)
    parser.add_argument('-n', '--node-count', default=1, type=int)
    parser.add_argument('-i', '--node-index', default=0, type=int)
    args = parser.parse_args()
    seed = args.seed
    processes = args.processes
    output_dir = args.output_dir
    node_count = args.node_count
    node_index = args.node_index

    if (node_index > node_count):
        print('invalid node index!')
        exit(0)

    signal.signal(signal.SIGINT, sigint)
    signal.signal(signal.SIGTERM, sigint)
    np.random.seed(seed)
    random.seed(seed)

    scenarios = get_scenarios_to_run(output_dir, node_count, node_index)
    verify(scenarios, output_dir, pool_size=processes)  

def compute_proof_statistics(results):
    totals = np.zeros((len(results[0][2]), 3), dtype=int)
    perfect = 0
    perfectly_unsafe = 0
    for result in tqdm(results):
        res = result[2]
        safe = get_safe(res)
        unsafe = get_unsafe(res)
        unknown = get_unknown(res)
        combined = np.array([safe, unsafe, unknown]).T
        totals += combined
        perfect += np.min(safe)
        perfectly_unsafe += np.min(unsafe)
    return totals / len(results), perfect / len(results), perfectly_unsafe / len(results)

def get_safe(safety):
    return np.array(np.array(safety) == True, dtype=int)

def get_unsafe(safety):
    return np.array(np.array(safety) == False, dtype=int)

def get_unknown(safety):
    return np.array(np.array(safety) == None, dtype=int)

def save_perfectly_unsafe(results, log_dir):
    for result in results:
        if np.min(1 - np.array(np.array(result[2]) == True, dtype=int)) > 0:
            save_scenario_results(result[0], result[1], result[2], log_dir)

def get_init(traces, index):
    return traces.root.sims[index][0][1:]

# def plot_reachtube(traces, var):
#     fig, ax = plt.subplots()
#     trace = extract_variable(traces, var)
#     x = np.arange(len(trace))
#     ax.vlines(x, trace[:, 0], trace[:, 1], colors='lightgray')
#     return fig, ax

# def plot_result_paper(result: Tuple[SimulationScenario, AnalysisTree, List[bool]]):
#     scenario, traces, safety = result
#     scenario.get_boluses()
#     fig, ax = plot_reachtube(traces, 'G')
#     sims = traces.root.sims
#     for i, sim in enumerate(sims):
#         y = extract_variable(sim, 'G', type=TraceType.SIM)
#         x = np.arange(len(y))
#         ax.plot(x, y, color='black')

#     ax.grid()
#     ax.set_xlabel('Time (min)')
#     ax.set_ylabel('Blood Glucose (mg/dL)')
#     return fig, ax

def plot_reachtube(traces, var, ax):
    trace = extract_variable(traces, var)
    x = np.arange(len(trace))
    ax.vlines(x, trace[:, 0], trace[:, 1], colors='silver')
    return ax

def plot_result_paper(result: Tuple[SimulationScenario, AnalysisTree, List[bool]]):
    scenario, traces, safety = result

    # Create stacked plots with shared x-axis
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 0.8], hspace=0.1)
    
    ax_main = fig.add_subplot(gs[0])
    ax_meals = fig.add_subplot(gs[1], sharex=ax_main)
    
    # --- Main glucose plot ---
    plot_reachtube(traces, 'G', ax_main)
    sims = traces.root.sims
    for sim in sims:
        y = extract_variable(sim, 'G', type=TraceType.SIM)
        x = np.arange(len(y)) # convert from minutes to hours
        ax_main.plot(x, y, color='black')

    
    upper = ax_main.get_ylim()[1]
    
    import matplotlib; matplotlib.rcParams['hatch.linewidth'] = 1

    ax_main.axhspan(0, 54, facecolor='red', alpha=0.5, hatch='//', edgecolor='black')  # severe hypo
    ax_main.axhspan(54, 70, facecolor='red', alpha=0.35)   # mild hypo
    ax_main.axhspan(70, 180, facecolor='green', alpha=0.15)  # target
    ax_main.axhspan(180, 250, facecolor='yellow', alpha=0.2)  # hyper
    ax_main.axhspan(250, upper, facecolor='orange', alpha=0.2)  # severe hyper
    ax_main.set_axisbelow(False)

    # Clear and reapply the limits (just in case hspans change them)
    ax_main.set_ylim(0, upper)
    
    ax_main.grid()
    ax_main.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
    ax_main.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_main.tick_params(axis='both', which='major', labelsize=12)

    # --- Meals plot (vertical strips for carb ranges) ---
    meal_times = [m.time for m in scenario.get_meals()]
    meal_ranges = [m.carbs for m in scenario.get_meals()]  # each is a (low, high) tuple

    for t, (low, high) in zip(meal_times, meal_ranges):
        rect_width = 5   # adjust width to match bolus dot "thickness"
        ax_meals.add_patch(Rectangle((t - rect_width/2, low),
                             rect_width, high - low,
                             color='tab:green'))

    ax_meals.set_yticks([0, 40, 80, 120, 160])
    ax_meals.set_ylim(0, 160)
    ax_meals.grid(axis='both')
    ax_meals.set_ylabel("Meal carbs (g)", fontsize=12)
    ax_meals.set_xticks(np.arange(0, 24 * 60 + 1, 60))
    ax_meals.set_xticklabels([str(i) for i in range(25)], fontsize=12)
    ax_meals.set_xlabel('Time (hours)', fontsize=12)
    ax_meals.tick_params(axis='both', which='major', labelsize=12)

    fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1, hspace=0.02)
    plt.margins(x=0, y=0)

    return fig, (ax_main, ax_meals)

def plot_results(result: Tuple[SimulationScenario, Any, Any]) -> go.Figure:
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

    scenario, traces, safety = result
    fig = plot_variable(traces, 'G', show=False)
    sims = traces.root.sims
    for i, sim in enumerate(sims):
        y = extract_variable(sim, 'G', type=TraceType.SIM)
        x = np.arange(len(y))
        # custom legend
        fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"trace {i}",
                marker=dict(color=colors[i]),
        ))
    y_mins = []
    y_maxs = []
    for trace_data in fig.data:
        y_mins.append(min(trace_data.y))
        y_maxs.append(max(trace_data.y))
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=12)))
    fig.update_layout(
    xaxis_title="Time (min)",
    yaxis_title="Blood Glucose (mg/dL)",
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=14)
        )
    )

    fig.update_layout(
        margin={'t':0,'l':0,'b':0,'r':0}
    )

    # reduce the horizontal whitespace in this image to a minimum
    # fig.update_layout(
    #     margin=dict(l=20, r=20),  # reduce left and right margins
    #     autosize=True,
    #     width=None  # let the renderer decide the width
    # )
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
    # run_verification_scenario(scenario)
    # init = get_init(scenario, 1)
    # print(init)
    # traces = simulate_from_init(scenario, init, logging=True, log_dir='results/logs')
    # plot_variable(traces, 'G')
    # results = load_results('results/verification')
    # print(unsafe_analysis(results, 2))
    # scenario, traces, safety = load_from_dir('results/perfectly_unsafe', 'scenario_0800000001ab5da68')
    # init = get_init(traces, 7)
    # print(init)
    # simulate_from_init(scenario, init, logging=True, log_dir='results/logs')
    # results = load_results('results/verification')
    # print(results)
    # titles = ['G < 54mg/dL for less than 1% of time', '54mg/dL <= G <= 70mg/dL for less than 4% of time', '70mg/dL <= G <= 180 mg/dL for at least 70% of time', '180mg/dL <= G <= 250 mg/dL for less than 25% of time', ' G > 250 mg/dL for < 5% of time']
    # for i in range(5):
    #     table_analysis(results, i, f'table_{i}', titles[i])

    verify_wrapper()