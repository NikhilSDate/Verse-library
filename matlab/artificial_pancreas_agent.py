import os, sys

from typing import List

import numpy as np
from tqdm import tqdm
from scipy.integrate import ode
from dotenv import load_dotenv
from cgm import CGM

load_dotenv()
EMULATOR_PATH = os.environ["EMULATOR_PATH"]
print(EMULATOR_PATH)
sys.path.insert(1, EMULATOR_PATH)
from pump_wrapper import Pump

from verse import BaseAgent, Scenario, ScenarioConfig
from verse.analysis.analysis_tree import TraceType, AnalysisTree

from artificial_pancreas_scenario import *
from hovorka_model import HovorkaModel
from pump_model import *
from state_utils import state_indices


# Combined human body system + insulin pump + scenario system


class ArtificialPancreasAgent(BaseAgent):

    def __init__(
        self,
        id,
        body: HovorkaModel,
        pump: InsulinPumpModel,
        cgm: CGM,
        simulation_scenario: SimulationScenario,
        code=None,
        file_name=None,
    ):

        super().__init__(id, code, file_name)

        self.body = body
        self.pump = pump
        self.cgm = cgm
        self.scenario = simulation_scenario
        

    def get_init_state(self, G):
        body_init_state = self.body.get_init_state(G)
        pump_init_state = self.pump.get_init_state()
        return list(body_init_state) + pump_init_state
        
    
    def get_init_range(self, Gl, Gh):
        lo, hi = self.body.get_init_range(Gl, Gh)
        real_lo = np.minimum(lo, hi)
        real_hi = np.maximum(lo, hi)
        pump_state = self.pump.get_init_state()
        return [list(real_lo) + pump_state, list(real_hi) + pump_state]
        
    def get_bg(self, Q1):
        return Q1 / self.body.hovorka_parameters()[12] * 18

    # TODO should mode be an enum?
    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:
        init = np.abs(init)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))

        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        state_vec = init

        carbs = 0
        meal_index = 0
        for i in tqdm(range(0, num_points)):

            state_vec[state_indices["G"]] = self.get_bg(state_vec[state_indices["Q1"]])
            current_time = i * time_step
            bolus, meal = self.scenario.get_events(current_time)

            bg = int(state_vec[state_indices['C']] * 18)
            self.cgm.post_reading(bg, current_time)
            bg = self.cgm.get_reading(current_time)
            # tqdm.write(f'bg({current_time}) = {bg}')
            carbs = 0
            if bolus:
                self.pump.send_bolus_command(bg, bolus)

            if meal:
                carbs = state_vec[state_indices[f'carbs_{meal_index}']]
                meal_index += 1
            dose = self.pump.pump_emulator.delay_minute(bg=bg)
            print(f'dose({i}) = {dose}')
            r = ode(lambda t, state: self.body.model(current_time + t, state, dose, carbs))
            r.set_initial_value(state_vec[:-1])
            res: np.ndarray = r.integrate(r.t + time_step)
            final = res.flatten()
            state_vec[:-1] = final
            state_vec[state_indices["iob"]] = self.pump.extract_state()[0]            
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = state_vec
        return trace
