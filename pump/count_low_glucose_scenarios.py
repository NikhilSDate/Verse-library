"""
Count how many verified scenarios in a results directory have a glucose (G)
reachtube lower bound that ever drops below a given threshold.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from artificial_pancreas_simulate import get_index
from verification import load_from_dir


def scenario_min_lower_bound(results_dir, scenario_dir, var_idx):
    """Return the minimum reachtube lower bound for var_idx, or None if no trace data."""
    _, traces, _ = load_from_dir(results_dir, scenario_dir)
    if traces is None:
        return None
    raw_rt = traces.root.trace.get('pump')
    if raw_rt is None:
        return None
    raw_rt = np.array(raw_rt)
    raw_rt = raw_rt.reshape(-1, 2, raw_rt.shape[1])
    lower_bound = raw_rt[:, 0, var_idx]
    return lower_bound.min()


def _worker(args):
    results_dir, scenario_dir, var_idx = args
    return scenario_min_lower_bound(results_dir, scenario_dir, var_idx)


def count_low_lower_bound(results_dir, threshold=30, variable='G', processes=1):
    var_idx = get_index(variable)
    scenario_dirs = sorted(
        d for d in os.listdir(results_dir)
        if d.startswith('scenario_') and os.path.isdir(os.path.join(results_dir, d))
    )

    total = len(scenario_dirs)
    below_threshold = 0
    missing = 0

    tasks = [(results_dir, scenario_dir, var_idx) for scenario_dir in scenario_dirs]
    if processes > 1:
        with Pool(processes) as pool:
            results = pool.imap(_worker, tasks, chunksize=16)
            for min_lower in tqdm(results, total=total):
                if min_lower is None:
                    missing += 1
                elif min_lower < threshold:
                    below_threshold += 1
    else:
        for task in tqdm(tasks):
            min_lower = _worker(task)
            if min_lower is None:
                missing += 1
            elif min_lower < threshold:
                below_threshold += 1

    return total, below_threshold, missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count scenarios whose glucose reachtube lower bound drops below a threshold'
    )
    parser.add_argument('results_dir', help='Results directory containing scenario_* subfolders')
    parser.add_argument('-t', '--threshold', type=float, default=30,
                        help='Lower-bound threshold to check against (default: 30)')
    parser.add_argument('-v', '--variable', default='G',
                        help='State variable to check (default: G)')
    parser.add_argument('-p', '--processes', type=int, default=24,
                        help='Number of worker processes (default: 24)')
    args = parser.parse_args()

    total, below_threshold, missing = count_low_lower_bound(
        args.results_dir, threshold=args.threshold, variable=args.variable, processes=args.processes
    )

    print(f'Total scenarios: {total}')
    print(f'Scenarios with {args.variable} lower bound < {args.threshold}: {below_threshold}')
    if missing:
        print(f'Scenarios with no trace data (excluded from the count above): {missing}')
