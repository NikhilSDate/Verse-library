import os, sys

from typing import List

import numpy as np
from tqdm import tqdm
from scipy.integrate import ode

from dotenv import load_dotenv

load_dotenv()
EMULATOR_PATH = os.environ["EMULATOR_PATH"]

from verse import BaseAgent, Scenario, ScenarioConfig
from verse.analysis.analysis_tree import TraceType, AnalysisTree

sys.path.insert(1, EMULATOR_PATH)
from pump_wrapper import Pump

from body_model import BodyModel
from pump_scenario import SimulationScenario, Bolus, BolusType
from pump_utils import (
    set_val,
    get_val,
    state_indices,
    num_continuous_variables,
    num_meals,
    state_variable_names,
    get_bg,
    extract_pump_state,
)  # work toward removing the state_indices import


class PumpAgent(BaseAgent):

    body = None

    def __init__(self, id, body_model, simulation_scenario, code=None, file_name=None):
        super().__init__(id, code, file_name)
        self.scenario: SimulationScenario = simulation_scenario
        self.body = body_model

    # TODO should mode be an enum
    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:

        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))

        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        pump = get_initialized_pump(init)
        meals = get_meals_from_state(init)
        dose = 0

        for i in tqdm(range(0, num_points)):
            current_time = i * time_step
            # init[state_indices["Isc1"]] += units_to_pmol_per_kg(dose)
            r = ode(lambda t, state: insulin_glucose_model(current_time + t, self.body, state, meals))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            init[0] = get_val(init, "Gp") / self.body.Vg
            bolus = self.scenario.get_bolus(current_time)
            if bolus:
                dose = handle_bolus(pump, init, bolus)
                tqdm.write(str(dose))
            elif current_time % 5 == 0:
                # TODO: need to actually model basal deliveries
                dose = self.scenario.basal_rate / 60 * 5  # basal rate = 0.5u/hr
            else:
                dose = 0
            pump.delay()
            extract_pump_state(init, pump)
            # pump.delay()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace


# TODO move to utils?
def get_initialized_pump(init, events=3, duration=4):
    # TODO allow different item durations
    iob_items = []
    for i in range(events):
        item_iob = get_val(init, f"pump_iob_{i}")
        item_elapsed = get_val(init, f"pump_elapsed_{i}")
        iob_items.append((item_iob, item_elapsed, duration))
    iob = get_val(init, "pump_iob")
    pump = Pump((iob_items, iob, duration))
    return pump


def insulin_glucose_model(t, body, state, meals):
    body_vars = state[:18]  # have 17 variables plus one ignore variable
    body_derivatives = body.model(t, body_vars, meals)

    pump_derivatives = np.zeros((num_continuous_variables - len(body_derivatives),))
    return np.concatenate([body_derivatives, pump_derivatives])


def get_meals_from_state(state):
    result = []
    for i in range(1, num_meals + 1):
        Di = get_val(state, f"D_{i}")
        ti = get_val(state, f"t_{i}")
        result.append((Di, ti))
    return result

# TODO if i put this in PumpAGent, i can just set the body object here instead of in constructor
def get_init_state(init_bg, meals):
    # TODO feed in BW
    BW = 78
    body = BodyModel(BW, init_bg)
    body_init_state = [
        init_bg,
        body.Gpb,
        body.Gtb,
        body.Ilb,
        body.Ipb,
        body.Ib,
        body.Ib,
        0,
        0,
        0,
        0,
        body.SRsHb,
        body.Hb,
        body.XHb,
        body.Isc1ss,
        body.Isc2ss,
        body.Hsc1ss,
        body.Hsc2ss,
    ]
    scenario_state = []
    for i in range(len(meals)):
        carbs = meals[i][0] * 1000  # convert g to mg
        scenario_state.append(carbs)
        scenario_state.append(meals[i][1])

    # TODO do this better
    # TODO make sure these values are being used in other places we mess with state
    pump_init_state = [
        0 for i in range(num_continuous_variables - len(body_init_state) - len(scenario_state))
    ]  # exclude D
    return body, body_init_state + scenario_state + pump_init_state


def handle_bolus(pump, state, bolus: Bolus) -> float:
    if bolus.type == BolusType.Simple:
        bg = get_bg(state)
        dose = pump.dose_simple(bg + 30, bolus.carbs)
        return dose
    else:
        # glucose = get_visible(init)[0]
        # carbs = time_to_carbs[current_time]
        # pump.dose_extended(glucose, carbs, 50, 120)
        # print('dosing')
        raise NotImplementedError()
