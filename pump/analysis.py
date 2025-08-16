import numpy as np
from typing import List, Tuple
from artificial_pancreas_scenario import SimulationScenario
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety.safety import AGP_report
from verification import *
from artificial_pancreas_simulate import extract_variable
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def get_scenario_AGP_reports(result: Tuple[SimulationScenario, AnalysisTree, List[bool]]):
    scenario, traces, safety = result
    glucose_reachtube = extract_variable(traces, 'G', type=TraceType.VERIF)
    reachtube_report = AGP_report(glucose_reachtube)
    sim_reports = []
    for sim in traces.root.sims:
        glucose_trace = extract_variable(sim, 'G', type=TraceType.SIM)
        sim_report = AGP_report(glucose_trace, type=TraceType.SIM)
        sim_reports.append(sim_report)
    return (reachtube_report, sim_reports)


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
    
# ranks scenarios by some key and save the top n
def rank_analysis(scenarios: Dict[SimulationScenario, Tuple[str, str]], key: Callable, n: int, output_dir: str, reverse=False):
    # sort scenarios by report value
    top_scenarios = sorted(list(scenarios.keys())[:n], key=lambda scenario: key(load_from_dir(*scenarios[scenario])), reverse=reverse)
    save_results_by_scenario(scenarios, top_scenarios[:10], output_dir)

def redzone_key(result: Tuple[SimulationScenario, AnalysisTree, List[bool]]):
    (reachtube_report, sim_reports) = get_scenario_AGP_reports(result)
    max_redzone_perc = -np.inf
    for sim_report in sim_reports:
        redzone = sim_report[0]
        max_redzone_perc = max(max_redzone_perc, redzone)
    return max_redzone_perc

def redzone_high_key(result: Tuple[SimulationScenario, AnalysisTree, List[bool]]):
    (reachtube_report, sim_reports) = get_scenario_AGP_reports(result)
    max_redzone_perc = -np.inf
    for sim_report in sim_reports:
        redzone = sim_report[-1]
        max_redzone_perc = max(max_redzone_perc, redzone)
    return max_redzone_perc 

def bad_verif_key(result: Tuple[SimulationScenario, AnalysisTree, List[bool]]):
    trace = result[1]
    glucose_trace = extract_variable(trace, 'G')
    diffs = glucose_trace[:, 1] - glucose_trace[:, 0]
    return max(diffs)

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

def table_analysis(results, zone, figname='table.png'):
    print('Starting table analysis')
    data = {}
    for result in tqdm(results):
        safety = result[2]
        key = (result[0].get_largest_meal(), result[0].get_total_carb_range()[1])
        if key not in data:
            data[key] = np.zeros((3,))
        data[key] += np.array([get_safe(safety)[zone], get_unsafe(safety)[zone], get_unknown(safety)[zone]])

    x_values = sorted(set(key[0] for key in data))
    y_values = sorted(set(key[1] for key in data), reverse=True)
    df = pd.DataFrame(index=y_values, columns=x_values, dtype=object)

    # Fill DataFrame with raw values (for drawing bars)
    for (x, y), vals in data.items():
        df.at[y, x] = vals

    # Replace NaNs with zero-triplets
    for i in df.index:
        for j in df.columns:
            if df.at[i, j] is None or isinstance(df.at[i,j], float) and np.isnan(df.at[i,j]):
                df.at[i,j] = np.array([0,0,0], dtype=float)


    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, len(x_values))
    ax.set_ylim(0, len(y_values))

    # Parameters
    colors = ['green', 'red', 'yellow']  # safe, unsafe, unknown
    width_frac = 0.85  # 70% of cell width and height
    height_frac = 0.7

    for yi, y in enumerate(y_values):
        for xi, x in enumerate(x_values):
            vals = data.get((x, y), np.zeros(3))
            total = np.sum(vals)

            # Grid cell (dotted gray border)
            ax.add_patch(plt.Rectangle(
                (xi, yi), 1, 1, fill=False, edgecolor='gray', lw=1, linestyle=':'
            ))

            if total > 0:
                # Compute bar dimensions (70% of cell, centered)
                bar_width = width_frac
                bar_height = height_frac
                x_offset = xi + (1 - width_frac) / 2
                y_offset = yi + (1 - height_frac) / 2

                start = 0
                for k, c in enumerate(colors):
                    frac = vals[k] / total
                    if frac > 0:
                        loc = (x_offset + start*bar_width, y_offset)
                        ax.text(loc[0] + bar_width * frac / 2, yi + 0.5, str(int(vals[k])), fontsize=9, ha='center', va='center', color='lightgray', weight='bold')
                        ax.add_patch(plt.Rectangle(
                            loc,
                            bar_width*frac, bar_height,
                            facecolor=c, edgecolor='none'
                        ))
                        start += frac
                            # Overlay total count in light gray
            ax.text(
                xi + 0.5, yi + 0.5,
                f"{int(total)}",
                ha='center', va='center',
                fontsize=9, color='lightgray', weight='bold'
            )

    legend_elements = [Patch(facecolor='green', label='Safe'), Patch(facecolor='red', label='Unsafe'), Patch(facecolor='yellow', label='Unknown')]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))

    # Set ticks as labels
    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    ax.set_xticklabels(x_values, fontsize=14)
    ax.set_yticklabels(y_values, fontsize=14)

    ax.invert_yaxis()  # top row = largest y
    ax.set_xlabel('Max single-meal carbs (g)', fontsize=14)
    ax.set_ylabel('Max total scenario carbs (g)', fontsize=14)

    # Hide spines for a cleaner table look
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join('./figures', figname), dpi=200)
    plt.show()

def all_zones_analysis(results: List[Tuple[SimulationScenario, AnalysisTree, List[bool]]]):
    safety_categories = ['Glucose region', 'Time-in-range safety criterion', 'Safe', 'Unsafe', 'Indeterminate']
    df = pd.DataFrame(columns=safety_categories)

    df.iloc[:, 0] = ['0 - 54 mg/dL', '54 - 70 mg/dL', '70 - 180 mg/dL', '180 - 250 mg/dL', '250+ mg/dL']
    df.iloc[:, 1] = [' < 1\%', '< 4\%', '> 70\% ', '< 25\%', '< 5\%']
    df = df.rename({'index': 'Glucose region'})
    df = df.fillna(0)
    count = 0
    for result in tqdm(results):
        safety = result[2]
        safe = get_safe(safety)
        unsafe = get_unsafe(safety)
        unknown = get_unknown(safety)
        df['Safe'] += safe
        df['Unsafe'] += unsafe
        df['Indeterminate'] += unknown

        count += 1

    df[['Safe', 'Unsafe', 'Indeterminate']] = df[['Safe', 'Unsafe', 'Indeterminate']] / count
    
    styler = df.style.format(
        {
            'Safe': lambda x: f"{x*100:.1f}\\%",
            'Unsafe': lambda x: f"{x*100:.1f}\\%",
            'Indeterminate': lambda x: f"{x*100:.1f}\\%"
        }
    )
    styler = styler.hide(axis='index')
    print(styler.to_latex(hrules=True))
    return df

def load_n_results(scenario_dir, n):
    g = load_results_gen(scenario_dir)
    results = []
    for i in tqdm(range(n)):
        results.append(next(g))
    return results



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
    # g = load_results_gen('/mnt/shared/gpfs/home/ndate2/InsulinPump/results/verification')
    # results = []
    # for i in tqdm(range(1000)):
    #     results.append(next(g))
    # table_analysis(results, 0)
    # breakpoint()

    # scenarios = load_scenarios_and_dirs('/mnt/shared/gpfs/home/ndate2/InsulinPump/results/verification')
    # rank_analysis(scenarios, bad_verif_key, 100, 'results/bad_verif', reverse=True)

    # results = load_results_gen('/mnt/shared/gpfs/home/ndate2/InsulinPump/results/verification')
    # all_zones_analysis(results)
    # plot_reachtube(traces, 'G')
    # fig = plot_results(results[0])
    # fig.write_image('test2.png')

    result = load_from_dir('./results/bad_verif', 'scenario_4')
    scenario, traces, safety = result
    print(hash(scenario))
    fig, ax = plot_result_paper(result)
    fig.savefig('./figures/extended_shutoff.png')

    # scenarios = load_scenarios_and_dirs('/mnt/shared/gpfs/home/ndate2/InsulinPump/results/verification')
    # rank_analysis(scenarios, key=redzone_high_key, n=5000, output_dir='results/redzone_high', reverse=True)
