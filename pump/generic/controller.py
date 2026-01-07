"""
Abstract Controller interface for cyber-physical system verification.

A Controller processes observable state from the plant and produces control outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class Controller(ABC):
    """
    Abstract base class for controllers in cyber-physical systems.

    A controller maintains internal state and computes control outputs based on
    observable plant state and external events (e.g., user inputs, setpoints).
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the controller's internal state.

        This is called at the start of each simulation/verification trace.
        The controller should reset its internal state (e.g., IOB, predictions)
        to initial values while keeping its configuration (settings, scenario).
        """
        pass

    @abstractmethod
    def step(self, state_vector: np.ndarray, time: float) -> Any:
        """
        Compute control output for one time step.

        The controller receives the full system state vector, which contains:
        [plant_state, controller_state, ...additional state components...]

        The controller extracts whatever information it needs from the state vector.

        Args:
            state_vector: Full system state vector (all state is observable)
            time: Current simulation time

        Returns:
            Control output to be applied to the plant
        """
        pass

    @abstractmethod
    def get_internal_state(self) -> np.ndarray:
        """
        Get the controller's internal state for verification purposes.

        This state is part of the full system state and may be used in
        verification specifications.

        Returns:
            Array containing controller's internal state variables
        """
        pass
