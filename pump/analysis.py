import numpy as np
from typing import List, Tuple
from artificial_pancreas_scenario import SimulationScenario
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety.safety import AGP_report
from verification import *
from artificial_pancreas_simulate import extract_variable

# we will store a map of scenario: ([], [])



def get_all_AGP_reports(scenarios: Dict[SimulationScenario, Tuple[str, str]]) -> Dict[SimulationScenario, Tuple]:
    reports = {}
    count = 0
    for scenario, path in scenarios.items():
        scenario, traces, safety = load_from_dir(path[0], path[1])
        glucose_reachtube = extract_variable(traces, 'G', type=TraceType.VERIF)
        reachtube_report = AGP_report(glucose_reachtube)
        sim_reports = []
        for sim in traces.root.sims:
            glucose_trace = extract_variable(sim, 'G', type=TraceType.SIM)
            sim_report = AGP_report(glucose_trace, type=TraceType.SIM)
            sim_reports.append(sim_report)
        reports[scenario] = (reachtube_report, sim_reports)
        count += 1
        if count % 100 == 0:
            print(count)
            if count == 1000:
                break
    return reports

def two_way_analysis(scenarios: List[Tuple[SimulationScenario, object, object]], index):
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

def compute_proof_statistics(results: List[Tuple[SimulationScenario, object, object]]):
    totals = np.zeros_like(results[0][2], dtype=int)
    perfect = 0
    perfectly_unsafe = 0
    for result in tqdm(results):
        totals += np.array(result[2], dtype=int)
        perfect += np.min(np.array(result[2], dtype=int))
        perfectly_unsafe += np.min(1 - np.array(result[2], dtype=int))
    return totals / len(results), perfect / len(results), perfectly_unsafe / len(results)

def save_results_by_scenario(all_scenarios: Dict[SimulationScenario, Tuple[str, str]], to_save: List[SimulationScenario], output_dir):
    for scenario in to_save:
        path = all_scenarios[scenario]
        result = load_from_dir(path[0], path[1])
        save_scenario_results(result[0], result[1], result[2], output_dir)
    

def redzone_analysis(scenarios: Dict[SimulationScenario, Tuple[str, str]]):
    def key(reports, scenario):
        report = reports[scenario]
        sim_reports = report[1]
        max_redzone_perc = 0
        for sim_report in sim_reports:
            redzone = sim_report[0]
            max_redzone_perc = max(max_redzone_perc, redzone)
        return max_redzone_perc
    
    reports = get_all_AGP_reports(scenarios)
    # sort scenarios by report value
    top_scenarios = sorted(reports.keys(), key=lambda scenario: key(reports, scenario), reverse=True)
    top_scenarios = set(top_scenarios[:10])
    breakpoint()
    save_results_by_scenario(scenarios, top_scenarios, 'results/bad_verif')

def extended_shutoff_analysis(result_func: Callable[[], Generator[Tuple[SimulationScenario, AnalysisTree, object], None, None]]):
   
    keys = {}
    for (scenario, traces, safety) in result_func():
        glucose_trace = extract_variable(traces, 'G')
        diffs = glucose_trace[:, 1] - glucose_trace[:, 0]
        keys[scenario] = max(diffs)
        pass
    
    scenarios = sorted(list(keys.keys()), key=keys.get, reverse=True)
    top_scenarios = set(scenarios[:10])
    save_results_by_scenario(result_func(), top_scenarios, 'results/bad_verif')

# walks through all subdirs in result_dir and converts any traces.pkl to traces.gzip
def compress_traces(result_dir):
    scenarios = []
    scenario_dirs = [ f for f in os.scandir(result_dir) if f.is_dir() ] if os.path.exists(result_dir) else []
    count = 0
    for scenario_dir in tqdm(scenario_dirs):
        trace_path = os.path.join(scenario_dir, 'traces.pkl')
        if not os.path.exists(trace_path):
            # compressed already
            continue
        
        try:
            with open(trace_path, 'rb') as f:
                traces = pickle.load(f)
        except:
            continue
        
        tqdm.write(f'compressing traces for {scenario_dir}')
        # dump gzip
        gzip_path = os.path.join(scenario_dir, 'traces.gzip')
        with gzip.open(gzip_path, 'wb') as f:
            pickle.dump(traces, f)    

def table_analysis(results: List[Tuple[SimulationScenario, object, object]], zone, figname='table.png', title='Table'):
    print('Starting table analysis')
    data = {}
    for result in tqdm(results):
        safety = result[2]
        key = (result[0].get_largest_meal(), result[0].get_total_carb_range()[1])
        if key not in data:
            data[key] = np.zeros((3,))
        data[key] += np.array([get_safe(safety)[zone], get_unsafe(safety[zone]), get_unknown(safety)[zone]])

    # convert to percentages


    x_values = sorted(set(key[0] for key in data))
    y_values = sorted(set(key[1] for key in data), reverse=True)
    df = pd.DataFrame(index=y_values, columns=x_values)

    # Fill DataFrame with formatted values
    for (x, y), vals in data.items():
        df.at[y, x] = f"[{int(vals[0])}, {int(vals[1])}, {int(vals[2])}]"
    df = df.fillna("--")
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    print(len(x_values))
    print(len(y_values))
    # Create table without row and column labels
    table = ax.table(
        cellText=df.values,
        cellLoc='center',
        loc='center', 
        rowLabels=y_values,
        colLabels=x_values
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Add axis labels
    plt.title(title)
    plt.figtext(0.5, 0.2, 'Max single-meal carbs', ha='center', va='center', fontsize=12)
    plt.figtext(0.02, 0.5, 'Max total carbs', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.savefig(figname)
    return data


if __name__ == '__main__':
    # debug_sim('results/bad_verif', 'scenario_5', 1)
    # seed = 42
    # f = open('hashes2.txt', 'w')
    # np.random.seed(seed)
    # random.seed(seed)
    # scenarios = gen_verification_scenarios()
    # np.random.shuffle(scenarios)
    # for scenario in scenarios:
    #     f.write(str(hash(scenario)) + '\n')
    # f.close()
    g = load_results_gen('/mnt/shared/gpfs/home/ndate2/InsulinPump/results/verification')
    results = []
    for i in tqdm(range(5000)):
        results.append(next(g))
    table_analysis(results, 0)
    breakpoint()