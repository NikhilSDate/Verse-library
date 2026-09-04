"""
Compute a reachtube around the average trace by:
  1. Loading simulation traces from a results folder.
  2. Treating trace[0] as the center; randomly sampling n_sample of the
     remaining traces to compute the variable's average. The traces NOT
     sampled (plus the center) are held out for fitting the reachtube, so
     the average and the reachtube are never computed from the same data.
  3. Appending an absolute-deviation column to the held-out traces
     (including the center).
  4. Fitting a DryVR-style reachtube to the augmented held-out traces.
  5. Plotting average ± deviation bound together with the raw traces.

The core functions (add_deviation_variable, compute_deviation_reachtube)
are completely agnostic to the state model and work on plain numpy arrays
of shape (N, T, ndims) where column 0 is time.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from artificial_pancreas_simulate import get_index
from verse.analysis.dryvr import get_reachtube_segment, calcDelta
from verification import load_from_dir


# ---------------------------------------------------------------------------
# Array-agnostic core
# ---------------------------------------------------------------------------

def add_deviation_variable(ref_traces, target_traces, var_idx, method='midpoint'):
    """
    Append an absolute-deviation column to target_traces.

    Computes a reference trajectory from ref_traces, then appends
    |trace[:, var_idx] - ref| as a new last column to every trace in
    target_traces (which may be a disjoint set of traces from ref_traces).

    Parameters
    ----------
    ref_traces    : ndarray (M, T, ndims)  – traces used to build the reference trajectory
    target_traces : ndarray (K, T, ndims)  – traces to augment with a deviation column
    var_idx       : int  – column index of the variable to track (includes time col)
    method        : 'midpoint' (default) or 'mean'
                    'midpoint' – (min + max) / 2 of ref_traces at each timestep
                    'mean'     – arithmetic mean of ref_traces

    Returns
    -------
    augmented : ndarray (K, T, ndims+1)
    ref_var   : ndarray (T,) – reference trajectory at var_idx
    """
    ref_vals = ref_traces[:, :, var_idx]
    if method == 'midpoint':
        ref_var = (ref_vals.min(axis=0) + ref_vals.max(axis=0)) / 2.0
    elif method == 'mean':
        ref_var = ref_vals.mean(axis=0)
    else:
        raise ValueError(f"method must be 'midpoint' or 'mean', got {method!r}")

    dev_col = np.abs(target_traces[:, :, var_idx] - ref_var)   # (K, T)
    augmented = np.concatenate(
        [target_traces, dev_col[:, :, np.newaxis]], axis=2
    )  # (K, T, ndims+1)

    return augmented, ref_var

def compute_deviation_reachtube(training_traces, var_idx, inits, n_sample=20, seed=None, method='midpoint'):
    """
    Fit a DryVR reachtube to traces augmented with an absolute-deviation variable.

    training_traces[0] is always the center trace. The remaining traces are
    split: n_sample of them are drawn at random to compute the reference
    average trajectory, and the rest (never overlapping with the sample) are
    held out, together with the center, to fit the reachtube. This keeps the
    average trace and the reachtube statistically independent of each other.

    Mirrors the logic in calc_bloated_tube exactly:
      – initial_radii = calcDelta = (max - min) / 2 of the trace range at t=0
      – get_reachtube_segment is called on the held-out augmented trace array

    Parameters
    ----------
    training_traces : ndarray (N, T, ndims)  – trace[0] is the center
    var_idx         : int  – column index of the variable of interest
    n_sample        : int  – number of non-center traces to sample for the average
    seed            : int or None  – random seed for the sample
    method          : 'midpoint' (default) or 'mean' – how to compute the reference

    Returns
    -------
    ref_var   : ndarray (T,)
    augmented : ndarray (K, T, ndims+1)  – held-out traces (center + rest), original vars + deviation column
    reachtube : ndarray (T-1, 2, ndims+1)
    """
    pool = training_traces[1:]
    n_pool = pool.shape[0]
    if n_sample >= n_pool:
        raise ValueError(
            f'n_sample ({n_sample}) must leave at least one trace for the reachtube '
            f'out of {n_pool} non-center traces'
        )

    rng = np.random.default_rng(seed)
    sample_mask = np.zeros(n_pool, dtype=bool)
    sample_mask[rng.choice(n_pool, size=n_sample, replace=False)] = True

    avg_traces = pool[sample_mask]
    held_out_traces = np.concatenate([training_traces[:1], pool[~sample_mask]], axis=0)

    augmented, ref_var = add_deviation_variable(avg_traces, held_out_traces, var_idx, method=method)
    dev_at_t0 = augmented[:, 0, -1]
    inits[0] = inits[0] + [-dev_at_t0.max()]
    inits[1] = inits[1] + [dev_at_t0.max()]
    initial_radii = calcDelta(inits[0], inits[1])
    initial_radii = np.array(initial_radii)
    reachtube = get_reachtube_segment(augmented, initial_radii, method='PW')
    return ref_var, augmented, reachtube


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_deviation_reachtube(
    avg_var,
    augmented_traces,
    reachtube,
    time_axis,
    var_idx,
    orig_reachtube=None,
    variable='G',
    output_path=None,
    show=True,
):
    """
    Plot average ± deviation reachtube with raw traces in the background.
    Optionally overlays the original reachtube.

    The deviation variable is the last column of reachtube / augmented_traces.

    Parameters
    ----------
    avg_var          : (T,)
    augmented_traces : (N, T, ndims+1)
    reachtube        : (T-1, 2, ndims+1)
    time_axis        : (T,)
    var_idx          : int  – column of the original variable (for raw traces)
    orig_reachtube   : (T', 2, ndims) or None – pre-computed reachtube reshaped
                       from (2*T', ndims); time at col 0, lower at [:,0,:], upper at [:,1,:]
    """
    dev_lo = reachtube[:, 0, -1]   # lower bound on deviation variable
    dev_hi = reachtube[:, 1, -1]   # upper bound on deviation variable
    t_reach = time_axis[1:]

    fig, (ax, ax_dev) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # ── Top: G space ──
    for trace in augmented_traces[1:]:
        ax.plot(time_axis, trace[:, var_idx], color='gray', alpha=0.25, linewidth=0.6)

    if orig_reachtube is not None:
        t_orig = orig_reachtube[:, 0, 0]
        ax.fill_between(
            t_orig,
            orig_reachtube[:, 0, var_idx],
            orig_reachtube[:, 1, var_idx],
            alpha=0.35,
            color='orange',
            label='Original reachtube',
        )

    ax.fill_between(
        t_reach,
        avg_var[1:] - dev_hi,
        avg_var[1:] + dev_hi,
        alpha=0.35,
        color='steelblue',
        label='Deviation reachtube',
    )
    ax.plot(time_axis, avg_var, color='steelblue', linewidth=2, label='Reference trace')
    ax.set_ylabel(f'{variable} (mg/dL)' if variable == 'G' else variable)
    ax.legend()
    ax.grid(True)

    # ── Bottom: deviation variable as-is ──
    for trace in augmented_traces[1:]:
        ax_dev.plot(time_axis, trace[:, -1], color='gray', alpha=0.25, linewidth=0.6)
    ax_dev.fill_between(t_reach, dev_lo, dev_hi, alpha=0.35, color='steelblue',
                        label='Deviation reachtube')
    ax_dev.set_xlabel('Time (min)')
    ax_dev.set_ylabel(f'{variable} deviation')
    ax_dev.legend()
    ax_dev.grid(True)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {output_path}')
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Top-level pipeline  (model-aware glue)
# ---------------------------------------------------------------------------

def run(output_dir, scenario_dir, n_sample=20, variable='G', seed=None, method='midpoint', output_path=None, show=True):
    """
    Full pipeline: load → compute → plot.

    Parameters
    ----------
    output_dir   : str  – parent results directory (e.g. 'results/verification2')
    scenario_dir : str  – sub-directory name (e.g. 'scenario_5050')
    n_sample     : int  – number of non-center traces to sample for the average
    variable     : str  – state variable to analyse (default 'G')
    seed         : int or None  – random seed for the sample
    output_path  : str or None  – where to save the plot (None = don't save)
    show         : bool – whether to call plt.show()
    """
    scenario, traces, safety = load_from_dir(output_dir, scenario_dir)
    if traces is None:
        raise FileNotFoundError(
            f'No traces found in {os.path.join(output_dir, scenario_dir)}'
        )

    # Build (N, T, ndims) array – trim to common length in case sims differ
    sims = traces.root.sims
    raw_inits = np.array(traces.root.init['pump'][0])   # (2, ndims)
    inits: list[list] = [raw_inits[0].tolist(), raw_inits[1].tolist()]
    min_len = min(s.shape[0] for s in sims)
    training_traces = np.array([s[:min_len] for s in sims])   # (N, T, ndims)

    var_idx = get_index(variable)   # only model-specific call
    time_axis = training_traces[0, :, 0]

    # Extract original reachtube: stored as (2*T', ndims), reshape to (T', 2, ndims)
    orig_rt = None
    raw_rt = traces.root.trace.get('pump')
    if raw_rt is not None:
        raw_rt = np.array(raw_rt)
        orig_rt = raw_rt.reshape(-1, 2, raw_rt.shape[1])

    avg_var, augmented, reachtube = compute_deviation_reachtube(
        training_traces, var_idx, inits=inits, n_sample=n_sample, seed=seed, method=method
    )

    fig = plot_deviation_reachtube(
        avg_var, augmented, reachtube, time_axis,
        var_idx=var_idx,
        orig_reachtube=orig_rt,
        variable=variable,
        output_path=output_path,
        show=show,
    )
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deviation reachtube for a scenario')
    parser.add_argument('output_dir',   help='Parent results directory')
    parser.add_argument('scenario_dir', help='Scenario sub-directory name')
    parser.add_argument('-n', '--n-sample', type=int, default=20,
                        help='Number of non-center traces to sample for the average; '
                             'the rest are held out for the reachtube (default: 20)')
    parser.add_argument('-v', '--variable', default='G',
                        help='State variable to analyse (default: G)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for trace sampling')
    parser.add_argument('-m', '--method', default='midpoint',
                        choices=['midpoint', 'mean'],
                        help='Reference trajectory method (default: midpoint)')
    parser.add_argument('-o', '--output', default=None,
                        help='Path to save the plot (default: show interactively)')
    args = parser.parse_args()

    run(
        args.output_dir,
        args.scenario_dir,
        n_sample=args.n_sample,
        variable=args.variable,
        seed=args.seed,
        method=args.method,
        output_path=args.output,
        show=(args.output is None),
    )
