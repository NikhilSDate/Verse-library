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
from state_utils import state_indices, num_meals


class Logger:
    def __init__(self, log_dir):
        self.dir = log_dir
        self.sim_idx = 0
        self.current_dose_file = None
        self.current_output_file = None
        self.output_buffer = [] # store lines of output from the pump
        self.t = 0
        
    def tick(self, increment=1):
        self.output_buffer.append('=' * 10 + f'time={self.t}' + '=' * 10)
        self.t += increment
        
    def start_sim(self):
        if self.current_dose_file is not None:
            self.current_dose_file.close()
        if self.current_output_file is not None:
            self.flush_all_output()                
            self.current_output_file.close()
        dose_path = os.path.join(self.dir, f'sim_{self.sim_idx}_dose.txt')
        output_path = os.path.join(self.dir, f'sim_{self.sim_idx}_output.txt')
        self.current_dose_file = open(dose_path, 'w+')
        self.current_output_file = open(output_path, 'w+')
        self.sim_idx += 1

    def write_dose(self, time, dose):
        if self.current_dose_file is None:
            print("Can't log without starting a simulation")
        self.current_dose_file.write(f'dose({time}) = {dose}\n')

    def flush_all_output(self):
        if self.current_output_file is None:
            return
        self.current_output_file.write('\n'.join(self.output_buffer))
        self.output_buffer.clear()
    
    def __del__(self):
        if self.current_dose_file is not None:
            self.current_dose_file.flush()
            self.current_dose_file.close()
        if self.current_output_file is not None:
            self.flush_all_output()
            self.current_output_file.flush()
            self.current_output_file.close()
        
        


# Combined human body system + insulin pump + scenario system


class ArtificialPancreasAgent(BaseAgent):

    def __init__(
        self,
        id,
        body: HovorkaModel,
        pump: InsulinPumpModel,
        cgm: CGM,
        simulation_scenario: SimulationScenario,
        logger: Logger,
        code=None,
        file_name=None,
    ):

        super().__init__(id, code, file_name)

        self.body = body
        self.pump = pump
        self.cgm = cgm
        self.scenario = simulation_scenario
        self.logger = logger
        self.pump.pump_emulator.link_output_buffer(logger.output_buffer)

        

    def get_init_state(self, G, meals):
        body_init_state = self.body.get_init_state(G)
        pump_init_state = self.pump.get_init_state()
        scenario_state = self.get_scenario_state()
        meal_state = self.get_meal_state(meals)
        return list(body_init_state) + pump_init_state + meal_state + scenario_state
    
    
    def get_meal_state(self, meals):
        meal_state  = [0] * num_meals
        for i, meal in enumerate(meals):
            meal_state[i] = meal.carbs
        return meal_state
            
    def get_scenario_state(self):
        return [0, 0]
    
    def get_init_range(self, Gl, Gh, ml, mh):
        lo, hi = self.body.get_init_range(Gl, Gh)
        real_lo = np.minimum(lo, hi)
        real_hi = np.maximum(lo, hi)
        pump_state = self.pump.get_init_state()
        meal_state_low = [0] * num_meals
        meal_state_high = [0] * num_meals
        for i in range(len(ml)):
            mli = np.minimum(ml[i].carbs, mh[i].carbs)
            mhi = np.maximum(ml[i].carbs, mh[i].carbs)
            meal_state_low[i] = mli
            meal_state_high[i] = mhi
        scenario_state = self.get_scenario_state()
        return [list(real_lo) + pump_state + meal_state_low + scenario_state, list(real_hi) + pump_state + meal_state_high + scenario_state]
        
    def get_bg(self, Q1):
        return Q1 / self.body.hovorka_parameters()[12] * 18
    
    def reset_pump(self):
        self.pump.reset_pump()
        self.pump.pump_emulator.link_output_buffer(self.logger.output_buffer)

    # TODO should mode be an enum?
    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:
        init = np.abs(init)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))

        self.reset_pump()
        self.logger.start_sim()
        
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        state_vec = init

        meal_index = 0
        
        predictions = [0] * num_points
        for i in tqdm(range(0, num_points)):
            
            self.logger.tick()

            state_vec[state_indices["G"]] = self.get_bg(state_vec[state_indices["Q1"]])
            current_time = i * time_step
            events: Tuple[Bolus, Meal] = self.scenario.get_events(current_time)
            bolus, meal = events
            bg = int(state_vec[state_indices['C']] * 18)
            self.cgm.post_reading(bg, current_time)
            bg = self.cgm.get_reading(current_time)
            carbs = 0
            
            # handle meal/bolus
            if meal:
                carbs = state_vec[state_indices[f'carbs_{meal_index}']]
                meal_index += 1


            if bolus and bolus.carbs == -1:
                # "fill in" carbs: later this should be moved to some "user agent"
                bolus.carbs = state_vec[state_indices[f'carbs_{meal_index}']]
            
            if bolus:
                self.pump.send_bolus_command(bg, bolus)

            dose = self.pump.pump_emulator.delay_minute(bg=bg)
            self.logger.write_dose(current_time, dose)
            
            r = ode(lambda t, state: self.body.model(current_time + t, state, dose, carbs))
            r.set_initial_value(state_vec[:self.body.num_variables])
            res: np.ndarray = r.integrate(r.t + time_step)
            final = res.flatten()
            
            # body state
            state_vec[:self.body.num_variables] = final
            
            # pump state
            pump_state = self.pump.extract_state()
            state_vec[state_indices["iob"]] = pump_state[0]         
            
            # scenario state is unchanged
            
            # derived state
            state_vec[state_indices["iob_error"]] = (state_vec[state_indices["iob"]] * 0.12 * 70 - state_vec[state_indices["I"]])
            prediction = pump_state[1]
            predictions[i] = prediction
            # prediction is 30 mins into the future
            
            # 30 min buffer for predictions to stabilize (this is more than necessary)
            if i >= 60 and predictions[i - 30] != -1:
                state_vec[state_indices["prediction_error"]] =  predictions[i - 30] - state_vec[state_indices['G']]
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = state_vec
        return trace
