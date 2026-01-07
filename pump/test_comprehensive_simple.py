"""
Simplified comprehensive test for the generic framework.

This test verifies that:
1. The generic framework produces physiologically plausible glucose trajectories
2. IOB tracking is correct
3. Controller and plant interactions work properly
4. Verification produces consistent results
"""

import numpy as np
from verification import gen_verification_scenarios
from artificial_pancreas_generic import simulate, verify
from state_utils import state_indices


def test_simulation_correctness():
    """Test that simulation produces correct and plausible results."""
    print("=" * 60)
    print("Testing simulation correctness...")
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

    print(f"\nTrace shape: {trace.shape}")
    print(f"Duration: {scenario.sim_duration} minutes")
    print(f"Number of timepoints: {trace.shape[0]}")

    # Extract key variables
    time_vec = trace[:, 0]
    G_idx = state_indices["G"] + 1
    glucose = trace[:, G_idx]
    iob_idx = state_indices["iob"] + 1
    iob = trace[:, iob_idx]

    # Check basic properties
    print(f"\nGlucose statistics:")
    print(f"  Initial: {glucose[0]:.1f} mg/dL")
    print(f"  Final: {glucose[-1]:.1f} mg/dL")
    print(f"  Min: {np.min(glucose):.1f} mg/dL")
    print(f"  Max: {np.max(glucose):.1f} mg/dL")
    print(f"  Mean: {np.mean(glucose):.1f} mg/dL")

    print(f"\nIOB statistics:")
    print(f"  Initial: {iob[0]:.3f} U")
    print(f"  Final: {iob[-1]:.3f} U")
    print(f"  Min: {np.min(iob):.3f} U")
    print(f"  Max: {np.max(iob):.3f} U")
    print(f"  Mean: {np.mean(iob):.3f} U")

    # Verification checks
    checks_passed = True

    # 1. Check glucose is physiologically plausible
    if np.any(glucose < 20):
        print("✗ Glucose went below 20 mg/dL (implausible)")
        checks_passed = False
    elif np.any(glucose > 600):
        print("✗ Glucose went above 600 mg/dL (implausible)")
        checks_passed = False
    else:
        print("✓ Glucose values are physiologically plausible")

    # 2. Check IOB is non-negative
    if np.any(iob < -0.001):  # Small tolerance for numerical errors
        print(f"✗ IOB went negative: min={np.min(iob):.3f}")
        checks_passed = False
    else:
        print("✓ IOB values are non-negative")

    # 3. Check for NaN or inf
    if np.any(np.isnan(trace)) or np.any(np.isinf(trace)):
        print("✗ Trace contains NaN or Inf values")
        checks_passed = False
    else:
        print("✓ No NaN or Inf values in trace")

    # 4. Check time vector is monotonic
    if not np.all(np.diff(time_vec) > 0):
        print("✗ Time vector is not monotonically increasing")
        checks_passed = False
    else:
        print("✓ Time vector is monotonically increasing")

    # 5. Check that glucose changes smoothly (no jumps > 50 mg/dL per minute)
    glucose_diff = np.abs(np.diff(glucose))
    max_jump = np.max(glucose_diff)
    if max_jump > 50:
        print(f"✗ Large glucose jump detected: {max_jump:.1f} mg/dL")
        checks_passed = False
    else:
        print(f"✓ Glucose changes smoothly (max change: {max_jump:.1f} mg/dL/min)")

    # 6. Check IOB decays over time if no insulin added
    # IOB should generally decrease or stay stable
    iob_increases = np.sum(np.diff(iob) > 0.1)  # Count significant increases
    if iob_increases > len(iob) * 0.3:  # More than 30% of time
        print(f"✗ IOB increases too frequently ({iob_increases} times)")
        checks_passed = False
    else:
        print(f"✓ IOB behavior is reasonable ({iob_increases} increases)")

    return checks_passed


def test_verification_correctness():
    """Test that verification produces consistent results."""
    print("\n" + "=" * 60)
    print("Testing verification correctness...")
    print("=" * 60)

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 120  # 2 hours for faster test

    # Run verification
    traces = verify(
        scenario=scenario,
        duration=scenario.sim_duration,
        time_step=1,
        num_simulations=5
    )

    # Check structure
    print(f"\nNumber of simulation traces: {len(traces.root.sims)}")

    checks_passed = True

    # 1. Check we got the expected number of simulations
    if len(traces.root.sims) < 4:  # Should be at least 4 (requested 5, but might get fewer)
        print(f"✗ Too few simulation traces: {len(traces.root.sims)}")
        checks_passed = False
    else:
        print(f"✓ Got {len(traces.root.sims)} simulation traces")

    # 2. Check each simulation trace
    for i, sim_trace in enumerate(traces.root.sims[:3]):  # Check first 3
        sim_array = np.array(sim_trace)
        if sim_array.shape[0] < 100:  # Should have at least 100 timepoints for 120 min
            print(f"✗ Simulation {i} too short: {sim_array.shape[0]} points")
            checks_passed = False
        if np.any(np.isnan(sim_array)) or np.any(np.isinf(sim_array)):
            print(f"✗ Simulation {i} contains NaN or Inf")
            checks_passed = False

    if checks_passed:
        print("✓ All simulation traces are valid")

    # 3. Check that simulations explore state space (not all identical)
    if len(traces.root.sims) >= 2:
        sim1 = np.array(traces.root.sims[0])
        sim2 = np.array(traces.root.sims[1])

        # Extract glucose from both
        G_idx = state_indices["G"] + 1
        glucose1 = sim1[:, G_idx]
        glucose2 = sim2[:, G_idx]

        # Check if they're different
        diff = np.max(np.abs(glucose1 - glucose2))
        if diff < 1.0:  # Less than 1 mg/dL difference
            print(f"✗ Simulations are too similar (max diff: {diff:.2f})")
            checks_passed = False
        else:
            print(f"✓ Simulations explore state space (max diff: {diff:.1f} mg/dL)")

    return checks_passed


def test_controller_plant_interaction():
    """Test that controller and plant interact correctly."""
    print("\n" + "=" * 60)
    print("Testing controller-plant interaction...")
    print("=" * 60)

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 180  # 3 hours

    # Run simulation
    traces = simulate(
        scenario=scenario,
        init_glucose=200,  # Start high to trigger insulin
        duration=scenario.sim_duration,
        time_step=1
    )

    trace = traces.root.trace['agent']

    G_idx = state_indices["G"] + 1
    glucose = trace[:, G_idx]
    iob_idx = state_indices["iob"] + 1
    iob = trace[:, iob_idx]

    checks_passed = True

    # 1. High glucose should eventually come down
    if glucose[-1] > glucose[0]:
        print(f"✗ Glucose increased from {glucose[0]:.1f} to {glucose[-1]:.1f}")
        checks_passed = False
    else:
        print(f"✓ Glucose decreased from {glucose[0]:.1f} to {glucose[-1]:.1f}")

    # 2. IOB should increase from basal (controller should deliver insulin for high BG)
    max_iob = np.max(iob)
    if max_iob < 0.5:  # Should have at least some IOB
        print(f"✗ IOB never increased significantly (max: {max_iob:.3f})")
        checks_passed = False
    else:
        print(f"✓ IOB increased appropriately (max: {max_iob:.3f} U)")

    # 3. Check that glucose responds to insulin (negative correlation with IOB)
    # Simple check: when IOB peaks, glucose should be decreasing
    iob_peak_idx = np.argmax(iob[10:]) + 10  # Skip first 10 min
    if iob_peak_idx < len(glucose) - 30:
        glucose_slope_after_peak = (glucose[iob_peak_idx + 30] - glucose[iob_peak_idx]) / 30
        if glucose_slope_after_peak > 0.5:  # Increasing after IOB peak
            print(f"✗ Glucose increasing after IOB peak ({glucose_slope_after_peak:.2f} mg/dL/min)")
            checks_passed = False
        else:
            print(f"✓ Glucose responding to insulin ({glucose_slope_after_peak:.2f} mg/dL/min)")

    return checks_passed


if __name__ == '__main__':
    try:
        # Run all tests
        results = []

        results.append(("Simulation correctness", test_simulation_correctness()))
        results.append(("Verification correctness", test_verification_correctness()))
        results.append(("Controller-plant interaction", test_controller_plant_interaction()))

        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        all_passed = True
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:35s} {status}")
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
