"""
Debug test to check if IOB is being computed correctly.
"""

import numpy as np
from verification import gen_verification_scenarios
from artificial_pancreas_generic import simulate
from state_utils import state_indices

# Run a simple simulation
scenarios = gen_verification_scenarios()
scenario = scenarios[0]
scenario.sim_duration = 60  # 1 hour

print("Running simulation...")
traces = simulate(
    scenario=scenario,
    init_glucose=200,  # High glucose to trigger insulin
    duration=scenario.sim_duration,
    time_step=1
)

trace = traces.root.trace['agent']

# Extract IOB
iob_idx = state_indices["iob"] + 1  # +1 for time column
iob = trace[:, iob_idx]

print(f"\nIOB values over time:")
print(f"Time | IOB")
for i in range(min(10, len(iob))):
    print(f"{i:4d} | {iob[i]:.4f}")

print(f"\n...")
print(f"Max IOB: {np.max(iob):.4f}")
print(f"Mean IOB: {np.mean(iob):.4f}")

# Check if pump is actually delivering insulin
# Let's also check what the controller is doing
print(f"\nChecking if controller exists in scenario...")
print(f"Scenario type: {type(scenario)}")
print(f"Scenario has user_config: {hasattr(scenario, 'user_config')}")

# Check meal/bolus schedule
print(f"\nScenario meals: {len(scenario.get_meals())}")
print(f"Scenario boluses: {len(scenario.get_boluses())}")

if len(scenario.get_boluses()) > 0:
    print("\nBolus schedule:")
    for b in scenario.get_boluses()[:3]:
        print(f"  Time {b.time}: {b.carbs} carbs, type={b.type}")
