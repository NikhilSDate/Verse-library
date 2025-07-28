import numpy as np
from typing import List, Tuple
from artificial_pancreas_scenario import SimulationScenario
import matplotlib.pyplot as plt
from tqdm import tqdm
from verification import *
from hashlib import md5
import json


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


def scenario_hash(scenario: SimulationScenario):
    dump = json.dumps(
        scenario.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
    )
    return md5(dump.encode('utf-8')).hexdigest()

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    scenarios = gen_verification_scenarios()
    np.random.shuffle(scenarios)
    results = load_results('results/test_local')
    existing = [result[0] for result in results]
    print(len(existing))
    print(len(set(existing)))
    breakpoint()