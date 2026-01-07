"""
Insulin Pump Controller implementing the generic Controller interface.

This controller wraps the insulin pump firmware emulator and implements the
Controller interface for use in the generic verification framework.
"""

import os
import sys
from dotenv import load_dotenv
import numpy as np
from typing import Any, Dict, Tuple
import pickle

load_dotenv()
EMULATOR_PATH = os.environ["EMULATOR_PATH"]
sys.path.insert(1, EMULATOR_PATH)
from pump_wrapper import Pump

from generic.controller import Controller
from artificial_pancreas_scenario import Bolus, BolusType, SimulationScenario
from state_utils import state_indices
import dataclasses


class InsulinPumpController(Controller):
    """
    Controller for insulin pump firmware.

    Observes full state vector containing glucose, meals, settings, etc.
    Control output: Insulin dose (float, in units)
    """

    def __init__(self, scenario: SimulationScenario = None, basal_iq: bool = True, trace: bool = False, output_buffer: list = None):
        """
        Initialize insulin pump controller.

        Args:
            scenario: Simulation scenario with meals, boluses, etc. (optional, can be set in init())
            basal_iq: Whether to enable basal IQ (automatic suspension)
            trace: Whether to trace pump calls for debugging
            output_buffer: Optional buffer for pump output logging
        """
        self.scenario = scenario
        self.basal_iq = basal_iq
        self.trace = trace
        self.output_buffer = output_buffer if output_buffer is not None else []
        self.settings = None
        self.pump_emulator = None

    def init(self, settings: Dict[str, Any], scenario: Any = None) -> None:
        """
        Initialize pump with settings.

        This should be called once when creating the controller.

        Args:
            settings: Dictionary with keys:
                - carb_ratio: Insulin-to-carb ratio
                - correction_factor: Correction factor for blood glucose
                - target_bg: Target blood glucose
                - max_bolus: Maximum bolus amount
                - insulin_duration: Insulin action duration
                - basal_rate: Basal insulin rate
            scenario: Simulation scenario (optional, can be set in __init__)
        """
        if scenario is not None:
            self.scenario = scenario
        self.settings = settings
        self.pump_emulator = Pump(basal_iq=self.basal_iq, trace=self.trace)
        self.pump_emulator.set_settings(
            carb_ratio=settings['carb_ratio'],
            correction_factor=settings['correction_factor'],
            target_bg=settings['target_bg'],
            max_bolus=settings['max_bolus'],
            insulin_duration=settings['insulin_duration'],
            basal_rate=settings['basal_rate']
        )
        self.pump_emulator.link_output_buffer(self.output_buffer)

    def reset(self) -> None:
        """Reset pump internal state to initial values."""
        if self.settings is not None:
            # Re-create pump emulator to reset state
            self.pump_emulator = Pump(basal_iq=self.basal_iq, trace=self.trace)
            self.pump_emulator.set_settings(
                carb_ratio=self.settings['carb_ratio'],
                correction_factor=self.settings['correction_factor'],
                target_bg=self.settings['target_bg'],
                max_bolus=self.settings['max_bolus'],
                insulin_duration=self.settings['insulin_duration'],
                basal_rate=self.settings['basal_rate']
            )
            self.pump_emulator.link_output_buffer(self.output_buffer)

    def step(self, state_vector: np.ndarray, time: float) -> float:
        """
        Compute insulin dose for one minute.

        Args:
            state_vector: Full system state vector
            time: Current time (minutes)

        Returns:
            Insulin dose (units) to deliver in this minute
        """
        # Extract CGM glucose reading from state vector
        bg = self._get_bg_from_state(state_vector)

        # Handle bolus events if scenario is available
        if self.scenario is not None:
            bolus, meal = self.scenario.get_events(time)
            if bolus:
                # Process bolus: fill in carbs from state vector if needed
                bolus_bg, processed_bolus = self._process_bolus(bolus, bg, state_vector)
                resume = self.scenario.user_config.resume if hasattr(self.scenario, 'user_config') else None
                self.send_bolus_command(bolus_bg, processed_bolus, resume)

        # Get dose from pump emulator
        dose = self.pump_emulator.step_minute(bg=bg)

        return dose

    def _get_bg_from_state(self, state_vector: np.ndarray) -> int:
        """Extract blood glucose reading from state vector."""
        # Get CGM reading from GluMeas
        return int(state_vector[state_indices['GluMeas']])

    def _process_bolus(self, bolus: Bolus, bg: int, state_vector: np.ndarray) -> Tuple[int, Bolus]:
        """Process bolus by filling in carbs from state vector if needed."""
        bolus_bg = bg if bolus else None

        if bolus and not bolus.carbs:
            # Fill in carbs from state vector
            carbs_raw = state_vector[state_indices[f'carbs_{bolus.meal_index}']]
            error = state_vector[state_indices[f'meal_{bolus.meal_index}_error']]
            carbs_errored = error * carbs_raw
            bolus = dataclasses.replace(bolus, carbs=carbs_errored)

        if bolus and not bolus.correction:
            bolus_bg = None

        return (bolus_bg, bolus)

    def send_bolus_command(self, bg: int, bolus: Bolus, resume: bool = None) -> None:
        """
        Send a bolus command to the pump.

        Args:
            bg: Blood glucose reading
            bolus: Bolus command
            resume: Whether to resume temp basal after bolus
        """
        if bolus.type == BolusType.Simple:
            self.pump_emulator.dose_simple(bg, bolus.carbs, resume=resume)
        else:
            self.pump_emulator.dose_extended(
                bg, bolus.carbs,
                bolus.config.deliver_now_perc,
                bolus.config.duration,
                resume=resume
            )

    def get_internal_state(self) -> np.ndarray:
        """
        Get pump internal state (IOB, prediction, etc.).

        Returns:
            Array with pump state: [iob, prediction, ...]
        """
        state = self.pump_emulator.get_state()
        return np.array([state[0]])  # Just IOB for now

    def write_trace(self, dir: str) -> None:
        """
        Write pump trace to directory.

        Args:
            dir: Directory to write trace files
        """
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'calls.pkl'), 'wb') as f:
            pickle.dump(self.pump_emulator.calls, f)

    @property
    def trace_metadata(self) -> Dict:
        """Get trace metadata from pump emulator."""
        return self.pump_emulator.trace_metadata


# Keep the original InsulinPumpModel for backward compatibility
class InsulinPumpModel:
    """
    Original InsulinPumpModel for backward compatibility.
    New code should use InsulinPumpController instead.
    """

    def __init__(self, sim_scenario, settings=None, trace=False):
        self.settings = settings
        self.basal_iq = settings['basal_iq']
        self.trace = trace
        self.pump_emulator = self.get_pump(self.basal_iq, settings, trace=trace)

    def reset_pump(self):
        self.pump_emulator = self.get_pump(self.basal_iq, self.settings, trace=self.trace)

    def get_pump(self, basal_iq, settings, trace=False):
        pump = Pump(basal_iq=basal_iq, trace=trace)
        if settings is not None:
            pump.set_settings(
                carb_ratio=settings['carb_ratio'],
                correction_factor=settings['correction_factor'],
                target_bg=settings['target_bg'],
                max_bolus=settings['max_bolus'],
                insulin_duration=settings['insulin_duration'],
                basal_rate=settings['basal_rate']
            )
        return pump

    def send_bolus_command(self, bg, bolus: Bolus, resume=False):
        if bolus.type == BolusType.Simple:
            self.pump_emulator.dose_simple(bg, bolus.carbs, resume=resume)
        else:
            self.pump_emulator.dose_extended(
                bg, bolus.carbs,
                bolus.config.deliver_now_perc,
                bolus.config.duration,
                resume=resume
            )

    def extract_state(self) -> Tuple[float]:
        state = self.pump_emulator.get_state()
        return state

    def get_init_state(self):
        return [0]

    def write_trace(self, dir):
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'calls.pkl'), 'wb') as f:
            pickle.dump(self.pump_emulator.calls, f)
