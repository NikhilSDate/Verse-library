import numpy as np
from typing import List, Tuple
from artificial_pancreas_scenario import SimulationScenario
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety.safety import AGP_report
from verification import *
from artificial_pancreas_simulate import extract_variable

# we will store a map of scenario: ([], [])

def get_all_AGP_reports(results_gen: Generator[Tuple[SimulationScenario, AnalysisTree, object], None, None]):
    reports = {}
    count = 0
    for (scenario, traces, safety) in results_gen:
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
    return reports

def two_way_analysis(results: List[Tuple[SimulationScenario, object, object]], index):
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

def save_results_by_scenario(results_gen, to_save, output_dir):
    to_save = set(to_save)
    breakpoint()
    for result in results_gen:
        if result[0] in to_save:
            save_scenario_results(result[0], result[1], result[2], output_dir)

def redzone_analysis(result_func: Callable[[], Generator[Tuple[SimulationScenario, AnalysisTree, object], None, None]], low: bool):
    def key(scenario):
        report = reports[scenario]
        return report[0][2][1] - report[0][2][0]
        sim_reports = report[1]
        max_redzone_perc = 0
        for sim_report in sim_reports:
            redzone = sim_report[0] if low else sim_report[-1]
            max_redzone_perc = max(max_redzone_perc, redzone)
        return max_redzone_perc
    
    reports = get_all_AGP_reports(result_func())
    # sort scenarios by report value
    scenarios = sorted(reports.keys(), key=key, reverse=True)
    top_scenarios = set(scenarios[:10])
    save_results_by_scenario(result_func(), top_scenarios, 'results/bad_verif')

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

if __name__ == '__main__':
    debug_sim('results/bad_verif', 'scenario_5', 1)
