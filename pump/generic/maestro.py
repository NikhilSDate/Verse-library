"""
Generic simulation/verification framework for controller-plant systems.

This module provides high-level simulate() and verify() functions that work with
any Controller and Plant implementations, handling all Verse integration internally.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from verse import Scenario, ScenarioConfig, BaseAgent
from verse.analysis.analysis_tree import AnalysisTree, TraceType

from .controller import Controller
from .plant import Plant
from tqdm import tqdm

class GenericControllerPlantAgent(BaseAgent):
    """
    Generic agent that integrates controller and plant without requiring subclassing.

    State vector structure: [plant_state, controller_state, scenario_state]
    - plant_state: Continuous dynamics evolved by plant.update()
    - controller_state: Controller internal state updated each step
    - scenario_state: Configuration/parameters that remain constant during simulation

    The entire state vector is observable to the controller.
    """

    def __init__(
        self,
        id: str,
        controller: Controller,
        plant: Plant,
        init_mode: Tuple = ("default",),
        preprocess_fn=None,
        code=None,
        file_name=None
    ):
        """
        Initialize generic controller-plant agent.

        Args:
            id: Agent identifier
            controller: Controller instance (already initialized)
            plant: Plant instance (already configured)
            init_mode: Initial discrete mode (default: ("default",))
            preprocess_fn: Optional function(state_vec, plant, time) to update derived state
            code: Optional code for Verse
            file_name: Optional file name for Verse
        """
        super().__init__(id, code, file_name)
        self.controller = controller
        self.plant = plant
        self.init_mode = init_mode
        self.preprocess_fn = preprocess_fn
        self.num_plant_vars = plant.get_num_variables()

        # Controller state size determined from initial state
        self.num_controller_vars = None  # Will be set on first step

    def TC_simulate(
        self,
        mode: list,
        init: np.ndarray,
        time_bound: float,
        time_step: float,
        lane_map=None
    ) -> TraceType:
        """
        Generic time-constrained simulation.

        This implements the standard controller-plant loop:
        1. Controller observes full state vector
        2. Controller computes control output
        3. Plant updates with control input
        4. Update plant and controller portions of state vector
        5. Scenario state remains unchanged

        Args:
            mode: Discrete mode (unused in generic implementation)
            init: Initial state vector [plant_state, controller_state, scenario_state]
            time_bound: Simulation duration
            time_step: Time step for discrete updates
            lane_map: Lane map (unused)

        Returns:
            Trace of state evolution over time
        """
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        state_vec = np.copy(init)

        # Reset controller at start of simulation
        self.controller.reset()

        # Determine controller state size and update initial state with controller state
        if self.num_controller_vars is None:
            controller_state = self.controller.get_internal_state()
            self.num_controller_vars = len(controller_state)

        # Update initial state vector with controller's initial state after reset
        controller_start = self.num_plant_vars
        controller_end = controller_start + self.num_controller_vars
        state_vec[controller_start:controller_end] = self.controller.get_internal_state()
        trace[0, 1:] = state_vec  # Update initial trace entry

        # Main simulation loop
        for i in tqdm(range(num_points)):
            current_time = i * time_step

            # Preprocess state (update derived quantities like sensor readings)
            if self.preprocess_fn is not None:
                self.preprocess_fn(state_vec, self.plant, current_time)

            # Controller observes entire state vector and computes control
            control = self.controller.step(state_vec, current_time)

            # Extract plant state and update with control
            plant_state = state_vec[:self.num_plant_vars]
            next_plant_state = self.plant.update(plant_state, control, current_time, time_step)

            # Update state vector (plant and controller portions only)
            # Scenario state (everything after controller state) remains unchanged
            state_vec[:self.num_plant_vars] = next_plant_state
            controller_start = self.num_plant_vars
            controller_end = controller_start + self.num_controller_vars
            state_vec[controller_start:controller_end] = self.controller.get_internal_state()

            # Store in trace
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = state_vec

        return trace


def _generate_state_class_code(num_vars: int) -> str:
    """Generate a minimal State class for Verse with the given number of variables."""
    var_lines = [f"    x{i}: float" for i in range(num_vars)]
    return """
import copy
from enum import Enum, auto

class Mode(Enum):
    default = auto()

class State:
""" + "\n".join(var_lines) + """
    agent_mode: Mode

    def __init__(self, x, agent_mode: Mode):
        pass

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output
"""


def simulate(
    controller: Controller,
    plant: Plant,
    init_state: np.ndarray,
    duration: float,
    time_step: float = 1.0
) -> AnalysisTree:
    """
    Run a single simulation with given controller and plant.

    Args:
        controller: Controller instance (already initialized)
        plant: Plant instance (already configured)
        init_state: Initial state vector [plant_state, controller_state, ...]
        duration: Simulation duration
        time_step: Time step for simulation

    Returns:
        AnalysisTree with simulation trace

    Example:
        >>> controller = MyController(...)
        >>> controller.init(settings, scenario)  # Initialize once
        >>> plant = MyPlant(...)
        >>> init_state = np.array([...])  # Full state vector
        >>> traces = simulate(controller, plant, init_state, duration=100, time_step=1)
    """
    # Generate State class code for Verse
    state_code = _generate_state_class_code(len(init_state))

    # Create generic agent (Verse internals hidden)
    agent = GenericControllerPlantAgent(
        id="agent",
        controller=controller,
        plant=plant,
        init_mode=("default",),
        preprocess_fn=None,
        code=state_code,
        file_name=None
    )

    # Create Verse scenario
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("agent", [init_state, init_state], ("default",))

    # Run simulation
    traces = scenario.simulate(duration, time_step)

    return traces


def verify(
    controller: Controller,
    plant: Plant,
    init_low: np.ndarray,
    init_high: np.ndarray,
    duration: float,
    time_step: float = 1.0,
    num_simulations: int = 10
) -> AnalysisTree:
    """
    Run verification with reachability analysis.

    Args:
        controller: Controller instance (already initialized)
        plant: Plant instance (already configured)
        init_low: Lower bounds on initial state vector
        init_high: Upper bounds on initial state vector
        duration: Simulation duration
        time_step: Time step for simulation
        num_simulations: Number of simulation traces for reachability

    Returns:
        AnalysisTree with reachable set and simulation traces

    Example:
        >>> controller = MyController(...)
        >>> controller.init(settings, scenario)  # Initialize once
        >>> plant = MyPlant(...)
        >>> init_low = np.array([...])   # Lower bounds on full state
        >>> init_high = np.array([...])  # Upper bounds on full state
        >>> traces = verify(controller, plant, init_low, init_high,
        ...                 duration=100, num_simulations=10)
    """
    # Generate State class code for Verse (use init_low for length)
    state_code = _generate_state_class_code(len(init_low))

    # Create generic agent (Verse internals hidden)
    agent = GenericControllerPlantAgent(
        id="pump",
        controller=controller,
        plant=plant,
        init_mode=("default",),
        preprocess_fn=None,
        code=state_code,
        file_name=None
    )

    # Create Verse scenario
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", [init_low, init_high], ("default",))

    # Run verification
    traces = scenario.verify(
        duration,
        time_step,
        params={'sim_trace_num': num_simulations, 'parallel': True}
    )

    return traces


# Backward compatibility: keep SimulationFramework class but it's deprecated
class SimulationFramework:
    """
    DEPRECATED: Use simulate() and verify() functions directly instead.

    This class is kept for backward compatibility but will be removed in future versions.
    """

    def __init__(self, controller_class, plant_class, state_builder, scenario_config):
        import warnings
        warnings.warn(
            "SimulationFramework class is deprecated. Use simulate() and verify() functions directly.",
            DeprecationWarning,
            stacklevel=2
        )
        self.controller_class = controller_class
        self.plant_class = plant_class
        self.state_builder = state_builder
        self.scenario_config = scenario_config


class StateBuilder:
    """
    DEPRECATED: State vectors should be built by users, not by framework.

    This class is kept for backward compatibility but will be removed in future versions.
    """

    def __init__(self):
        import warnings
        warnings.warn(
            "StateBuilder class is deprecated. Build state vectors directly in your code.",
            DeprecationWarning,
            stacklevel=2
        )

    def get_init_state(self, init_params: Any) -> np.ndarray:
        raise NotImplementedError()

    def get_init_range(self, init_low: Any, init_high: Any) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def get_initial_mode(self) -> Tuple:
        raise NotImplementedError()
