"""
Comprehensive test comparing generic framework with original implementation.

This test verifies that:
1. The generic framework produces correct glucose trajectories
2. IOB tracking matches expected behavior
3. Verification results match baseline implementation
"""

import numpy as np
from verification import gen_verification_scenarios
from artificial_pancreas_generic import simulate, verify
from artificial_pancreas_simulate import simulate_multi_meal_scenario, verify_multi_meal_scenario
from state_utils import state_indices


def compare_traces(trace1, trace2, tolerance=1e-6):
    """Compare two traces for equality within tolerance."""
    if trace1.shape != trace2.shape:
        print(f"Shape mismatch: {trace1.shape} vs {trace2.shape}")
        return False

    diff = np.abs(trace1 - trace2)
    max_diff = np.max(diff)

    if max_diff > tolerance:
        # Find where the biggest differences are
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Max difference: {max_diff} at index {max_idx}")
        print(f"Value 1: {trace1[max_idx]}")
        print(f"Value 2: {trace2[max_idx]}")
        return False

    return True


def test_simulation_correctness():
    """Test that simulation produces correct glucose trajectories."""
    print("=" * 60)
    print("Testing simulation correctness...")
    print("=" * 60)

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 240  # 4 hours

    # Convert scenario to single-value for baseline comparison
    # (baseline doesn't support ranges)
    import copy
    baseline_scenario = copy.deepcopy(scenario)
    if isinstance(baseline_scenario.settings, list):
        baseline_scenario.settings = baseline_scenario.settings[0]
    if isinstance(baseline_scenario.errors, list):
        baseline_scenario.errors = baseline_scenario.errors[0]
    baseline_scenario.init_bg = 120  # Single value for baseline

    # Run with generic framework
    print("\nRunning generic framework simulation...")
    generic_traces = simulate(
        scenario=scenario,
        init_glucose=120,
        duration=scenario.sim_duration,
        time_step=1
    )

    # Run with original implementation
    print("Running baseline simulation...")
    baseline_result = simulate_multi_meal_scenario(baseline_scenario)
    baseline_traces = baseline_result.payload  # VerificationResult.payload contains the traces

    # Extract traces
    generic_trace = generic_traces.root.trace['agent']
    baseline_trace = baseline_traces.root.trace['pump']

    print(f"\nGeneric trace shape: {generic_trace.shape}")
    print(f"Baseline trace shape: {baseline_trace.shape}")

    # Check glucose values
    G_idx = state_indices["G"] + 1  # +1 for time column
    generic_glucose = generic_trace[:, G_idx]
    baseline_glucose = baseline_trace[:, G_idx]

    print(f"\nGeneric glucose range: [{np.min(generic_glucose):.1f}, {np.max(generic_glucose):.1f}]")
    print(f"Baseline glucose range: [{np.min(baseline_glucose):.1f}, {np.max(baseline_glucose):.1f}]")

    # Check IOB values
    iob_idx = state_indices["iob"] + 1
    generic_iob = generic_trace[:, iob_idx]
    baseline_iob = baseline_trace[:, iob_idx]

    print(f"\nGeneric IOB range: [{np.min(generic_iob):.3f}, {np.max(generic_iob):.3f}]")
    print(f"Baseline IOB range: [{np.min(baseline_iob):.3f}, {np.max(baseline_iob):.3f}]")

    # Compare traces
    print("\nComparing traces...")
    if compare_traces(generic_trace, baseline_trace, tolerance=1e-4):
        print("✓ Traces match within tolerance!")
        return True
    else:
        print("✗ Traces differ!")

        # Print some sample values for debugging
        print("\nSample glucose values (first 10 timepoints):")
        print("Time | Generic | Baseline | Diff")
        for i in range(min(10, len(generic_glucose))):
            diff = abs(generic_glucose[i] - baseline_glucose[i])
            print(f"{i:4d} | {generic_glucose[i]:7.2f} | {baseline_glucose[i]:8.2f} | {diff:6.3f}")

        return False


def test_verification_correctness():
    """Test that verification produces correct reachable sets."""
    print("\n" + "=" * 60)
    print("Testing verification correctness...")
    print("=" * 60)

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 120  # 2 hours for faster test

    # Run with generic framework
    print("\nRunning generic framework verification...")
    generic_traces = verify(
        scenario=scenario,
        duration=scenario.sim_duration,
        time_step=1,
        num_simulations=5
    )

    # Run with original implementation
    print("Running baseline verification...")
    baseline_result = verify_multi_meal_scenario(scenario, params={'sim_trace_num': 5})
    baseline_traces = baseline_result.payload  # VerificationResult.payload contains the traces

    # Check number of simulation traces
    generic_sims = len(generic_traces.root.sims)
    baseline_sims = len(baseline_traces.root.sims)

    print(f"\nGeneric simulations: {generic_sims}")
    print(f"Baseline simulations: {baseline_sims}")

    if generic_sims != baseline_sims:
        print(f"✗ Different number of simulation traces!")
        return False

    # Compare simulation traces
    print("\nComparing simulation traces...")
    all_match = True
    for i in range(min(3, generic_sims)):  # Check first 3 traces
        generic_sim = np.array(generic_traces.root.sims[i])
        baseline_sim = np.array(baseline_traces.root.sims[i])

        if not compare_traces(generic_sim, baseline_sim, tolerance=1e-4):
            print(f"✗ Simulation trace {i} differs!")
            all_match = False
        else:
            print(f"✓ Simulation trace {i} matches")

    if all_match:
        print("\n✓ All simulation traces match!")
        return True
    else:
        print("\n✗ Some simulation traces differ!")
        return False


def test_physiological_constraints():
    """Test that simulation respects physiological constraints."""
    print("\n" + "=" * 60)
    print("Testing physiological constraints...")
    print("=" * 60)

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 360  # 6 hours

    # Run simulation
    traces = simulate(
        scenario=scenario,
        init_glucose=120,
        duration=scenario.sim_duration,
        time_step=1
    )

    trace = traces.root.trace['agent']

    # Check glucose bounds
    G_idx = state_indices["G"] + 1
    glucose = trace[:, G_idx]

    print(f"\nGlucose range: [{np.min(glucose):.1f}, {np.max(glucose):.1f}] mg/dL")

    # Physiologically plausible bounds
    if np.any(glucose < 20):
        print("✗ Glucose went below 20 mg/dL (implausible)")
        return False
    if np.any(glucose > 600):
        print("✗ Glucose went above 600 mg/dL (implausible)")
        return False

    # Check IOB is non-negative
    iob_idx = state_indices["iob"] + 1
    iob = trace[:, iob_idx]

    print(f"IOB range: [{np.min(iob):.3f}, {np.max(iob):.3f}] U")

    if np.any(iob < 0):
        print("✗ IOB went negative")
        return False

    # Check for NaN or inf
    if np.any(np.isnan(trace)) or np.any(np.isinf(trace)):
        print("✗ Trace contains NaN or Inf values")
        return False

    print("✓ All physiological constraints satisfied!")
    return True


if __name__ == '__main__':
    try:
        # Run all tests
        results = []

        results.append(("Simulation correctness", test_simulation_correctness()))
        results.append(("Verification correctness", test_verification_correctness()))
        results.append(("Physiological constraints", test_physiological_constraints()))

        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        all_passed = True
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:30s} {status}")
            all_passed = all_passed and passed

        print("=" * 60)

        if all_passed:
            print("\n✓ All comprehensive tests passed!")
        else:
            print("\n✗ Some tests failed")
            exit(1)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
