"""
Artificial pancreas simulation using the generic controller-plant framework.

This module provides high-level functions to run insulin pump verification
using the generic framework, hiding all Verse-specific details.
"""

import numpy as np
from typing import Tuple

import generic.maestro as maestro
from insulin_pump_controller import InsulinPumpController
from hovorka_plant import HovorkaPlant
from cgm import CGM
from artificial_pancreas_scenario import SimulationScenario, get_meal_range
from artificial_pancreas_simulate import get_cgm_error_range
from state_utils import num_meals
from verse.analysis.analysis_tree import AnalysisTree


def build_init_state(
    scenario: SimulationScenario,
    controller: InsulinPumpController,
    plant: HovorkaPlant,
    init_glucose: float
) -> np.ndarray:
    """
    Build initial state vector from scenario and initial glucose.

    Args:
        scenario: Simulation scenario
        controller: Controller instance
        plant: Plant instance
        init_glucose: Initial glucose level (mg/dL)

    Returns:
        Full initial state vector
    """
    # Get plant initial state (12 Hovorka variables: G through GluMeas)
    # plant.get_init_state() already returns state with G and GluMeas computed
    plant_state = plant.get_init_state(init_glucose)

    # Get controller initial state
    controller_state = controller.get_internal_state()

    # Build meal state
    meal_state = [0.0] * num_meals
    for i, meal in enumerate(scenario.get_meals()):
        # Handle tuple ranges by taking first value
        carbs = meal.carbs[0] if isinstance(meal.carbs, tuple) else meal.carbs
        meal_state[i] = float(carbs)

    # Build scenario state (derived, settings, errors, cgm)
    iob_error = 1.0
    prediction_error = 0.0
    settings = scenario.settings if not isinstance(scenario.settings, list) else scenario.settings[0]
    basal_rate = float(settings['basal_rate'])

    # Meal errors
    errors = scenario.errors if not isinstance(scenario.errors, list) else scenario.errors[0]
    if np.ndim(errors) == 0:
        meal_errors = [float(errors)] * num_meals
    else:
        meal_errors = [float(e) for e in errors]

    # CGM config
    cgm_a = float(scenario.cgm_config.bias if not isinstance(scenario.cgm_config.bias, tuple) else scenario.cgm_config.bias[0])
    cgm_b = float(scenario.cgm_config.offset if not isinstance(scenario.cgm_config.offset, tuple) else scenario.cgm_config.offset[0])

    # Combine all parts (matching State class order in verse_model.py)
    init_state = (
        list(plant_state) +          # 12 Hovorka variables (G through GluMeas)
        list(controller_state) +      # 1 controller variable (iob)
        meal_state +                  # 10 meal carbs
        [iob_error, prediction_error] +  # 2 derived state
        [basal_rate] +                # 1 settings
        meal_errors +                 # 10 meal errors
        [cgm_a, cgm_b]                # 2 CGM config
    )

    return np.array(init_state)


def build_init_range(
    scenario: SimulationScenario,
    controller: InsulinPumpController,
    plant: HovorkaPlant
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build initial state range for verification.

    Args:
        scenario: Simulation scenario with ranges
        controller: Controller instance
        plant: Plant instance

    Returns:
        Tuple of (init_low, init_high) state vectors
    """
    # Get ranges from scenario
    init_glucose_low, init_glucose_high = scenario.init_bg
    settings_low, settings_high = scenario.settings
    errors_low, errors_high = scenario.errors
    meals_low, meals_high = get_meal_range(scenario.get_meals())
    cgm_low, cgm_high = get_cgm_error_range(scenario.cgm_config)

    # Plant state range (12 Hovorka variables: G through GluMeas)
    # plant.get_init_range() already returns states with G and GluMeas computed
    plant_low, plant_high = plant.get_init_range(init_glucose_low, init_glucose_high)

    # Controller state (same for both)
    controller_state = controller.get_internal_state()

    # Meal state ranges (pad to num_meals with 0.0)
    meal_state_low = [0.0] * num_meals
    meal_state_high = [0.0] * num_meals
    for i, m in enumerate(meals_low):
        meal_state_low[i] = min(m.carbs) if isinstance(m.carbs, tuple) else m.carbs
    for i, m in enumerate(meals_high):
        meal_state_high[i] = max(m.carbs) if isinstance(m.carbs, tuple) else m.carbs

    # Derived state (same for both)
    iob_error = 1.0
    prediction_error = 0.0

    # Settings ranges
    basal_rate_low = settings_low['basal_rate']
    basal_rate_high = settings_high['basal_rate']

    # Error ranges
    if np.ndim(errors_low) == 0:
        meal_errors_low = [errors_low] * num_meals
        meal_errors_high = [errors_high] * num_meals
    else:
        meal_errors_low = list(errors_low)
        meal_errors_high = list(errors_high)

    # Build low and high state vectors (matching State class order)
    init_low = np.array(
        list(plant_low) +           # 12 Hovorka variables (G through GluMeas)
        list(controller_state) +    # 1 controller variable (iob)
        meal_state_low +            # 10 meal carbs
        [iob_error, prediction_error] +  # 2 derived state
        [basal_rate_low] +          # 1 settings
        meal_errors_low +           # 10 meal errors
        cgm_low                     # 2 CGM config
    )

    init_high = np.array(
        list(plant_high) +          # 12 Hovorka variables (G through GluMeas)
        list(controller_state) +    # 1 controller variable (iob)
        meal_state_high +           # 10 meal carbs
        [iob_error, prediction_error] +  # 2 derived state
        [basal_rate_high] +         # 1 settings
        meal_errors_high +          # 10 meal errors
        cgm_high                    # 2 CGM config
    )

    return (init_low, init_high)


def simulate(
    scenario: SimulationScenario,
    init_glucose: float,
    duration: float,
    time_step: float = 1.0,
    log_dir: str = None
) -> AnalysisTree:
    """
    Run a single artificial pancreas simulation.

    Args:
        scenario: Simulation scenario
        init_glucose: Initial glucose level (mg/dL)
        duration: Simulation duration (minutes)
        time_step: Time step (minutes)
        log_dir: Optional directory for logging

    Returns:
        AnalysisTree with simulation trace
    """
    # Create components
    cgm = CGM()
    plant = HovorkaPlant(scenario.params, cgm)
    plant.set_meals(scenario.get_meals())
    plant.set_cgm_config(scenario.cgm_config)

    settings = scenario.settings if not isinstance(scenario.settings, list) else scenario.settings[0]
    controller = InsulinPumpController(
        scenario=scenario,
        basal_iq=settings['basal_iq'],
        trace=False,
        output_buffer=[]
    )
    # Initialize controller once
    controller.init(settings, scenario)

    # Build initial state with initialized controller
    init_state = build_init_state(scenario, controller, plant, init_glucose)

    # Run simulation using generic framework (controller already initialized)
    traces = maestro.simulate(
        controller=controller,
        plant=plant,
        init_state=init_state,
        duration=duration,
        time_step=time_step
    )

    return traces


def verify(
    scenario: SimulationScenario,
    num_simulations: int = 10,
    log_dir: str = None
) -> AnalysisTree:
    """
    Run verification for artificial pancreas.

    Args:
        scenario: Simulation scenario with ranges
        duration: Simulation duration (minutes)
        time_step: Time step (minutes)
        num_simulations: Number of simulation traces
        log_dir: Optional directory for logging

    Returns:
        AnalysisTree with reachable set and simulation traces
    """
    # Create components
    cgm = CGM()
    plant = HovorkaPlant(scenario.params, cgm)
    plant.set_meals(scenario.get_meals())
    plant.set_cgm_config(scenario.cgm_config)

    settings = scenario.settings[0] if isinstance(scenario.settings, list) else scenario.settings
    controller = InsulinPumpController(
        scenario=scenario,
        basal_iq=settings['basal_iq'],
        trace=True,
        output_buffer=[]
    )
    # Initialize controller once
    controller.init(settings, scenario)

    # Build initial state range with initialized controller
    init_low, init_high = build_init_range(scenario, controller, plant)

    # Run verification using generic framework (controller already initialized)
    traces = maestro.verify(
        controller=controller,
        plant=plant,
        init_low=init_low,
        init_high=init_high,
        duration=scenario.sim_duration,
        time_step=scenario.time_step,
        num_simulations=num_simulations
    )

    return traces
