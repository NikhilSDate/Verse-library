"""
Test the new generic framework with artificial pancreas.
"""

from verification import gen_verification_scenarios
from artificial_pancreas_generic import simulate, verify

def test_generic_simulate():
    """Test generic simulation."""
    print("Testing generic simulate()...")

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 60  # 1 hour for quick test

    traces = simulate(
        scenario=scenario,
        init_glucose=120,
        duration=scenario.sim_duration,
        time_step=1
    )

    print(f"Simulation completed!")
    print(f"Trace shape: {traces.root.trace['agent'].shape}")
    return traces


def test_generic_verify():
    """Test generic verification."""
    print("\nTesting generic verify()...")

    scenarios = gen_verification_scenarios()
    scenario = scenarios[0]
    scenario.sim_duration = 60  # 1 hour for quick test

    traces = verify(
        scenario=scenario,
        duration=scenario.sim_duration,
        time_step=1,
        num_simulations=3  # Reduced for quick test
    )

    print(f"Verification completed!")
    # For verification, trace may be a list
    trace = traces.root.trace['agent']
    if hasattr(trace, 'shape'):
        print(f"Trace shape: {trace.shape}")
    else:
        print(f"Trace type: {type(trace)}, length: {len(trace)}")
    print(f"Number of simulation traces: {len(traces.root.sims)}")
    return traces


if __name__ == '__main__':
    try:
        # Test simulation first
        sim_traces = test_generic_simulate()
        print("✓ Simulation test passed\n")

        # Test verification
        verify_traces = test_generic_verify()
        print("✓ Verification test passed")

        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
