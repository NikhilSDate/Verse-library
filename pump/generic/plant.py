"""
Abstract Plant interface for cyber-physical system verification.

A Plant represents the physical system being controlled, maintaining continuous
state and evolving according to differential equations.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Plant(ABC):
    """
    Abstract base class for plants (physical systems) in cyber-physical systems.

    A plant maintains continuous state and evolves according to differential
    equations, responding to control inputs from the controller.
    """

    @abstractmethod
    def get_num_variables(self) -> int:
        """
        Get the number of state variables in the plant model.

        Returns:
            Number of continuous state variables
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get the current plant state.

        Returns:
            Current state vector
        """
        pass

    @abstractmethod
    def get_observable_state(self, state: np.ndarray, time: float) -> Any:
        """
        Extract observable portion of state that the controller can see.

        This typically includes sensor readings, measurements, etc. that may
        differ from the true state due to sensor noise, delays, or other factors.

        Args:
            state: Full plant state vector
            time: Current simulation time

        Returns:
            Observable state (format depends on the specific plant/controller)
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, control: Any, time: float, time_step: float) -> np.ndarray:
        """
        Update plant state given control input over a time step.

        This method integrates the differential equations governing the plant
        dynamics from time t to t + time_step, with the given control input.

        Args:
            state: Current plant state vector
            control: Control input from controller
            time: Current simulation time
            time_step: Integration time step

        Returns:
            Updated plant state vector after time_step
        """
        pass

    @abstractmethod
    def get_init_state(self, init_params: Any) -> np.ndarray:
        """
        Compute initial plant state from initialization parameters.

        Args:
            init_params: Parameters specifying the initial condition
                        (format depends on the specific plant)

        Returns:
            Initial plant state vector
        """
        pass

    @abstractmethod
    def get_init_range(self, init_params_low: Any, init_params_high: Any) -> tuple:
        """
        Compute range of initial plant states for verification.

        Args:
            init_params_low: Lower bound on initialization parameters
            init_params_high: Upper bound on initialization parameters

        Returns:
            Tuple of (lower_bound_state, upper_bound_state)
        """
        pass
