import numpy as np
from typing import List, Tuple
from artificial_pancreas_scenario import SimulationScenario
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety.safety import AGP_report
from verification import *
from artificial_pancreas_simulate import extract_variable

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

def redzone_analysis(results: List[Tuple[SimulationScenario, object, object]], low: bool):
    def key(result):
        traces = result[1]
        sims = traces.root.sims
        max_redzone_perc = 0
        for sim in sims:
            glucose_trace = extract_variable(sim, 'G', type=TraceType.SIM)
            report = AGP_report(glucose_trace, type=TraceType.SIM)
            redzone = report[0] if low else report[-1]
            max_redzone_perc = max(max_redzone_perc, redzone)
        return max_redzone_perc
    
    results = sorted(results, key=key, reverse=False)
    out_dir = 'results/redzone_low' if low else 'results/redzone_high'
    out_dir = 'results/test'
    for i in range(20):
        save_result_with_sims(results[i], out_dir)


if __name__ == '__main__':
    results = load_results('results/verification')
    redzone_analysis(results, low=True)