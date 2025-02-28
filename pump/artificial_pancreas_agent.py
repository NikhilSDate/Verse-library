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
from simutils import FORGOT_BOLUS

import dataclasses

# "logger" that actually does not do any logging
class NotLogger:
    def __init__(self):
        self.output_buffer = []
        pass
    def tick(self, increment=1):
        pass
    def start_sim(self):
        pass
    def write_dose(self, time, dose):
        pass

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

    # exclude from picling
    # TODO: we can probably make this more fine-grainged
    def __getstate__(self):
        return ()
    
    # exclude from picling
    # TODO: we can probably make this more fine-grainged
    def __setstate__(self, state):
        pass    

    def get_init_state(self, G, meals, settings):
        body_init_state = self.body.get_init_state(G)
        pump_init_state = self.pump.get_init_state()
        scenario_state = self.get_scenario_state()
        meal_state = self.get_meal_state(meals)
        settings_state = self.get_settings_state(settings)
        return list(body_init_state) + pump_init_state + meal_state + scenario_state + settings_state
    
    
    def get_meal_state(self, meals):
        meal_state  = [0] * num_meals
        for i, meal in enumerate(meals):
            meal_state[i] = meal.carbs
        return meal_state
            
    def get_scenario_state(self):
        return [0, 0]
    
    def get_settings_state(self, settings):
        return [settings['basal_rate']]
    
    def get_init_range(self, Gl, Gh, ml, mh, sl, sh):
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
        settings_low = self.get_settings_state(sl)
        settings_high = self.get_settings_state(sh)
        return [list(real_lo) + pump_state + meal_state_low + scenario_state + settings_low, list(real_hi) + pump_state + meal_state_high + scenario_state + settings_high]
        
    def get_bg(self, Q1):
        return Q1 * 18 / self.body.param['Vg']
    
    def reset_pump(self):
        self.pump.reset_pump()
        self.pump.pump_emulator.link_output_buffer(self.logger.output_buffer)

    def get_meals(self, state_vec):
        raw_meals = self.scenario.get_meals()
        meals = []
        for i, orig_meal in enumerate(raw_meals):
            carbs = state_vec[state_indices[f'carbs_{i}']]
            meals.append(dataclasses.replace(orig_meal, carbs=carbs))
        return meals
    
    
    
    # TODO should mode be an enum?
    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:
        init = np.abs(init)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))

        self.reset_pump()
        basal_rate = init[state_indices['basal_rate']]
        self.pump.pump_emulator.set_settings(basal_rate=basal_rate)
        
        
        self.logger.start_sim()
        
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        state_vec = init

        self.body.set_meals(self.get_meals(state_vec))
                
        predictions = [0] * num_points
        meal_index = 0
    
        
        for i in tqdm(range(0, num_points)):
            
            self.logger.tick()

            state_vec[state_indices["G"]] = self.get_bg(state_vec[state_indices["GluPlas"]])
            
            GluMeas = self.body.mmol_to_mgdl(state_vec[state_indices["GluInte"]])

            current_time = i * time_step
            events: Tuple[Bolus, Meal] = self.scenario.get_events(current_time)
            bolus, meal = events
            bg = int(GluMeas)
            self.cgm.post_reading(bg, current_time)
            bg = self.cgm.get_reading(current_time)
            
            state_vec[state_indices['GluMeas']] = bg
             
            carbs = 0
            
            # handle meal/bolus
            if meal:
                carbs = state_vec[state_indices[f'carbs_{meal_index}']]
                meal_index += 1


            if bolus:
                if bolus.carbs == -1:
                # "fill in" carbs: later this should be moved to some "user agent"
                    bolus = dataclasses.replace(bolus, carbs=carbs)
                    self.pump.send_bolus_command(bg, bolus)
                elif bolus.carbs == FORGOT_BOLUS:
                    if meal_index == 0:
                        raise ValueError("Forgot Bolus but no meal")
                    carbs = state_vec[state_indices[f'carbs_{meal_index - 1}']]
                    bolus = dataclasses.replace(bolus, carbs=carbs)
                    self.pump.send_bolus_command(None, bolus)
                else:
                    self.pump.send_bolus_command(bg, bolus)
                

            dose = self.pump.pump_emulator.delay_minute(bg=bg)
            
            if dose < 1 and bg < 60:
                dose = 0
            
            self.logger.write_dose(current_time, dose)
            
            r = ode(lambda t, state: self.body.model(current_time + t, state, dose))
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
            # state_vec[state_indices["iob_error"]] = (state_vec[state_indices["iob"]] * 0.12 * 70 - state_vec[state_indices["I"]])
            prediction = pump_state[1]
            predictions[i] = prediction
            # prediction is 30 mins into the future
            
            # 30 min buffer for predictions to stabilize (this is more than necessary)
            if i >= 60 and predictions[i - 30] != -1:
                state_vec[state_indices["prediction_error"]] =  predictions[i - 30] - state_vec[state_indices['G']]
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = state_vec
        return trace
