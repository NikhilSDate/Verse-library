import os, sys

from typing import List

import numpy as np
from tqdm import tqdm
from scipy.integrate import ode
from dotenv import load_dotenv

load_dotenv()
EMULATOR_PATH = os.environ["EMULATOR_PATH"]
print(EMULATOR_PATH)
sys.path.insert(1, EMULATOR_PATH)
from pump_wrapper import Pump

from verse import BaseAgent, Scenario, ScenarioConfig
from verse.analysis.analysis_tree import TraceType, AnalysisTree

from artificial_pancreas_scenario import *
from body_model import BodyModel
from pump_model import *
from state_utils import state_indices


# Combined human body system + insulin pump + scenario system


class ArtificialPancreasAgent(BaseAgent):

    def __init__(
        self,
        id,
        body: BodyModel,
        pump: InsulinPumpModel,
        simulation_scenario: SimulationScenario,
        code=None,
        file_name=None,
    ):

        super().__init__(id, code, file_name)

        self.body = body
        self.pump = pump
        self.scenario = simulation_scenario

    def get_init_state(self):

        init_state = [
            self.body.Gb,
            self.body.Gpb,
            self.body.Gtb,
            self.body.Ilb,
            self.body.Ipb,
            self.body.Ib,
            self.body.Ib,
            0,
            0,
            0,
            0,
            self.body.SRsHb,
            self.body.Hb,
            self.body.XHb,
            self.body.Isc1ss,
            self.body.Isc2ss,
            self.body.Hsc1ss,
            self.body.Hsc2ss,
        ]

        return init_state

    # TODO should mode be an enum?
    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:

        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))

        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        state_vec = init

        carbs = 0

        for i in tqdm(range(0, num_points)):

            current_time = i * time_step
            bolus, meal = self.scenario.get_events(current_time)

            
            dose = 0
            if bolus:
                bg = state_vec[state_indices['G']]
                dose = self.pump.send_bolus_command(bg, bolus)
                print(dose)
                # state_vec["Isc1"] += units_to_pmol_per_kg(dose)

            if meal:
                carbs = meal.carbs
                # TODO make sure this handles multiple carb inputs correctly
            
            self.pump.pump_emulator.delay()
            
            state_vec[state_indices['Isc1']] += units_to_pmol_per_kg(dose)


            r = ode(lambda t, state: self.body.model(current_time + t, state, carbs))
            r.set_initial_value(state_vec)
            res: np.ndarray = r.integrate(r.t + time_step)
            state_vec = res.flatten()


            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = state_vec

        return trace
