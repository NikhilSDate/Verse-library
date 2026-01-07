"""
Hovorka glucose-insulin model implementing the generic Plant interface.

This plant wraps the Hovorka physiological model and implements the Plant
interface for use in the generic verification framework.
"""

import numpy as np
from scipy.integrate import ode
from typing import Any, Tuple

from generic.plant import Plant
from hovorka_model import HovorkaModel
from cgm import CGM
from artificial_pancreas_scenario import Meal
from state_utils import state_indices


class HovorkaPlant(Plant):
    """
    Plant model for human glucose-insulin dynamics using the Hovorka model.

    State: 12-variable state including Hovorka differential equations + derived quantities
           [G, InsSub1, InsSub2, InsPlas, InsActT, InsActD, InsActE, GutAbs, GluPlas, GluComp, GluInte, GluMeas]
    Observable state: CGM glucose reading (with noise/bias)
    Control input: Insulin dose (units/minute)
    """

    def __init__(self, param: dict, cgm: CGM = None):
        """
        Initialize Hovorka plant.

        Args:
            param: Hovorka model parameters
            cgm: Continuous glucose monitor (for observable state)
        """
        self.model = HovorkaModel(param)
        self.cgm = cgm if cgm is not None else CGM()
        # Plant state includes G (derived) + 10 differential vars + GluMeas (derived) = 12 total
        self.num_variables = 12

    def get_num_variables(self) -> int:
        """Get number of state variables (12: includes derived G and GluMeas)."""
        return self.num_variables

    def get_state(self) -> np.ndarray:
        """
        Get current plant state.

        Returns:
            11-element state vector for Hovorka model
        """
        # This method is not used in the current implementation since state
        # is passed explicitly to update(), but included for interface completeness
        raise NotImplementedError("State is managed externally in current implementation")

    def get_observable_state(self, state: np.ndarray, time: float) -> int:
        """
        Get CGM glucose reading from plant state.

        Args:
            state: Full state vector (includes Hovorka state + other components)
            time: Current time

        Returns:
            CGM glucose reading (mg/dL, integer)
        """
        # Extract glucose from interstitial compartment (GluInte)
        GluMeas = self.model.mmol_to_mgdl(state[state_indices["GluInte"]])
        bg_raw = int(GluMeas)

        # Apply CGM noise/bias
        bg = self.cgm.get_reading(bg_raw)

        return bg

    def update(self, state: np.ndarray, control: float, time: float, time_step: float) -> np.ndarray:
        """
        Update Hovorka model state with insulin control.

        Args:
            state: Current Hovorka state (12 variables: G, InsSub1-GluInte, GluMeas)
            control: Insulin dose (units)
            time: Current time (minutes)
            time_step: Integration time step (minutes)

        Returns:
            Updated Hovorka state (12 variables) with derived quantities updated
        """
        # The Hovorka model uses 1-indexed arrays where model.eInsSub1=1, ..., model.eGluInte=10
        # We need to create an 11-element array for the ODE solver
        # Our state has: [G(0), InsSub1(1), ..., GluInte(10), GluMeas(11)] - 12 elements total
        # Extract [InsSub1, ..., GluInte] and prepend a dummy element for 1-indexing
        diff_state_for_ode = np.zeros(11)
        diff_state_for_ode[1:11] = state[1:11]  # Copy InsSub1 through GluInte to indices 1-10

        # Integrate the Hovorka differential equations
        r = ode(lambda t, state_: self.model.model(time + t, state_, control))
        r.set_initial_value(diff_state_for_ode)
        res = r.integrate(r.t + time_step)
        integrated = res.flatten()

        # Build full 12-element state with updated values
        result = np.zeros(12)
        result[:] = state[:]  # Copy all 12 elements
        result[1:11] = integrated[1:11]  # Update InsSub1 through GluInte from integrated result

        # Compute derived quantities
        # G (blood glucose in mg/dL) from GluPlas (index 8 in our state)
        GluPlas = result[state_indices["GluPlas"]]
        result[state_indices["G"]] = self.get_bg(GluPlas)

        # GluMeas (CGM reading) from GluInte (index 10 in our state)
        GluInte = result[state_indices["GluInte"]]
        result[state_indices["GluMeas"]] = self.model.mmol_to_mgdl(GluInte)

        return result

    def get_init_state(self, init_params: float) -> np.ndarray:
        """
        Get initial state from initial glucose level.

        Args:
            init_params: Initial glucose level (mg/dL)

        Returns:
            Initial Hovorka state vector
        """
        return self.model.get_init_state(init_params)

    def get_init_range(self, init_params_low: float, init_params_high: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get range of initial states for verification.

        Args:
            init_params_low: Lower bound on initial glucose (mg/dL)
            init_params_high: Upper bound on initial glucose (mg/dL)

        Returns:
            Tuple of (lower_bound_state, upper_bound_state)
        """
        return self.model.get_init_range(init_params_low, init_params_high)

    def set_meals(self, meals: list) -> None:
        """
        Set meal plan for the simulation.

        Args:
            meals: List of Meal objects
        """
        self.model.set_meals(meals)

    def set_cgm_config(self, config) -> None:
        """
        Set CGM configuration (bias, offset).

        Args:
            config: CGMConfig object
        """
        self.cgm.set_config(config)

    def mmol_to_mgdl(self, G: float) -> float:
        """Convert glucose from mmol/L to mg/dL."""
        return self.model.mmol_to_mgdl(G)

    def mgdl_to_mmol(self, G: float) -> float:
        """Convert glucose from mg/dL to mmol/L."""
        return self.model.mgdl_to_mmol(G)

    def get_bg(self, Q1: float) -> float:
        """
        Get blood glucose from plasma glucose compartment.

        Args:
            Q1: Glucose in plasma compartment (GluPlas)

        Returns:
            Blood glucose (mg/dL)
        """
        return Q1 * 18 / self.model.param['Vg']
