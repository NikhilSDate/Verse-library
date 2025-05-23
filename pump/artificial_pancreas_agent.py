import os, sys

from typing import List

import numpy as np
from tqdm import tqdm
from scipy.integrate import ode
from dotenv import load_dotenv
from cgm import CGM
import multiprocessing

load_dotenv()
EMULATOR_PATH = os.environ["EMULATOR_PATH"]
sys.path.insert(1, EMULATOR_PATH)
from pump_wrapper import Pump

from verse import BaseAgent, Scenario, ScenarioConfig
from verse.analysis.analysis_tree import TraceType, AnalysisTree

from artificial_pancreas_scenario import *
from hovorka_model import HovorkaModel
from pump_model import *
from state_utils import state_indices, num_meals, state_get, state_set
from simutils import FORGOT_BOLUS

import dataclasses

# "logger" that actually does not do any logging
class NotLogger:
    def __init__(self):
        self.output_buffer = []
        pass
    def start_sim(self, init):
        pass
    def step(self, dose, step=1):
        pass
    def error_dump(self):
        return self.output_buffer
    
    def get_output_buffer(self):
        return self.output_buffer

class Logger:
    def __init__(self, log_dir):
        self.dir = log_dir
        self.sim_idx = 0
        self.current_dose_file = None
        self.current_output_file = None
        self.output_buffer = [] # store lines of output from the pump
        self.t = 0
        self.init = None
        
    def step(self, dose, step=1):
        self.output_buffer.append('=' * 10 + f'time={self.t}' + '=' * 10)
        self.t += step
        if self.current_dose_file is not None:
            self.current_dose_file.write(f'dose({self.t}) = {dose}\n')
        
    def start_sim(self, init):
        self.flush()
        if self.dir is not None:
            dose_path = os.path.join(self.dir, f'sim_{self.sim_idx}_dose.txt')
            output_path = os.path.join(self.dir, f'sim_{self.sim_idx}_output.txt')
            self.current_dose_file = open(dose_path, 'w+')
            self.current_output_file = open(output_path, 'w+')
        self.sim_idx += 1
        self.init = init

    def flush(self):
        if self.current_dose_file is not None:
            self.current_dose_file.close()
        if self.current_output_file is not None:
            self.current_output_file.write('\n'.join(self.output_buffer))
            self.current_output_file.close()
        self.output_buffer.clear()

    def __del__(self):
        if self.current_dose_file is not None:
            self.current_dose_file.flush()
            self.current_dose_file.close()
        if self.current_output_file is not None:
            self.flush()
            self.current_output_file.flush()
            self.current_output_file.close()

    def get_output_buffer(self):
        return self.output_buffer

    def error_dump(self):
        return ErrorInfo(self.init, '\n'.join(self.output_buffer)) # get any unflushed data

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
        self.inits = []

    # exclude from picling
    # TODO: we can probably make this more fine-grainged
    def __getstate__(self):
        return ()
    
    # exclude from picling
    # TODO: we can probably make this more fine-grainged
    def __setstate__(self, state):
        pass    

    def get_init_state(self, G, meals, settings, errors):
        body_init_state = self.body.get_init_state(G)
        pump_init_state = self.pump.get_init_state()
        scenario_state = self.get_scenario_state()
        meal_state = self.get_meal_state(meals)
        settings_state = self.get_settings_state(settings)
        error_state = self.get_error_state(errors, num_meals)
        return list(body_init_state) + pump_init_state + meal_state + scenario_state + settings_state + list(error_state)
    
    
    def get_meal_state(self, meals):
        meal_state  = [0] * num_meals
        for i, meal in enumerate(meals):
            meal_state[i] = meal.carbs
        return meal_state
            
    def get_scenario_state(self):
        return [0, 0]
    
    def get_settings_state(self, settings):
        return [settings['basal_rate']]  
    
    def get_error_state(self, errors, num_meals):
        if errors is None:
            return [1] * num_meals
        if np.ndim(errors) == 0:
            return np.repeat(errors, num_meals)
        return errors
    
    def get_init_range(self, Gl, Gh, ml, mh, sl, sh, el, eh, cgm_low, cgm_high):
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
        
        assert(np.ndim(eh) == np.ndim(el))

        errors_low = self.get_error_state(el, num_meals)
        errors_high = self.get_error_state(eh, num_meals)
        
        return [list(real_lo) + pump_state + meal_state_low + scenario_state + settings_low + list(errors_low) + cgm_low, list(real_hi) + pump_state + meal_state_high + scenario_state + settings_high + list(errors_high) + cgm_high]
        
    def get_bg(self, Q1):
        return Q1 * 18 / self.body.param['Vg']
    
    def reset_pump(self):
        self.pump.reset_pump()
        self.pump.pump_emulator.link_output_buffer(self.logger.get_output_buffer())

    def get_meals(self, state_vec):
        raw_meals = self.scenario.get_meals()
        meals = []
        for i, orig_meal in enumerate(raw_meals):
            carbs = state_vec[state_indices[f'carbs_{i}']]
            meals.append(dataclasses.replace(orig_meal, carbs=carbs))
        return meals
    
    def process_bolus(self, bolus, bg, state_vec):
        if bolus:
            bolus_bg = bg
        if not bolus.carbs:
            # "fill in" carbs: later this should be moved to some "user agent"
            carbs_raw = state_vec[state_indices[f'carbs_{bolus.meal_index}']]
            error = state_vec[state_indices[f'meal_{bolus.meal_index}_error']]
            carbs_errored = error * carbs_raw
            bolus = dataclasses.replace(bolus, carbs=carbs_errored)
        if not bolus.correction:
            bolus_bg = None
            
        return (bolus_bg, bolus)
    
    # for any potential modifications to the pump's logic
    def apply_custom_logic(self, state_vec, dose):
        pass
    
    def get_config(self, state_vec):
        return CGMConfig(bias=state_get(state_vec, 'cgm_config_a'), offset=state_get(state_vec, 'cgm_config_b'))
    
    def apply_analyses(self, time, state_vec):
        # setup
        if not hasattr(self.apply_analyses, 'predictions'):
            self.apply_analyses.predictions = []
        
        predictions: List[float] = self.apply_analyses.predictions
        
        # pump state
        pump_state = self.pump.extract_state()
        state_vec[state_indices["iob"]] = pump_state[0]         
        
        # scenario state is unchanged
        
        # derived state
        # state_vec[state_indices["iob_error"]] = (state_vec[state_indices["iob"]] * 0.12 * 70 - state_vec[state_indices["I"]])
        prediction = pump_state[1]
        predictions.append(prediction)
        # prediction is 30 mins into the future
        
        # 30 min buffer for predictions to stabilize (this is more than necessary)
        if time >= 60 and predictions[int(time) - 30] != -1:
            state_vec[state_indices["prediction_error"]] =  self.predictions[i - 30] - state_vec[state_indices['G']]
    
    
    # TODO should mode be an enum?
    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        state_vec = init
        self.reset_pump()
        basal_rate = init[state_indices['basal_rate']]
        self.pump.pump_emulator.set_settings(basal_rate=basal_rate)
        self.logger.start_sim(init)
        self.body.set_meals(self.get_meals(state_vec))
        self.cgm.set_config(self.get_config(state_vec))
        
        process = multiprocessing.current_process()
        position = process._identity[0] if len(process._identity) > 0 else 0
        for i in tqdm(range(0, num_points), position=position):
            state_vec[state_indices["G"]] = self.get_bg(state_vec[state_indices["GluPlas"]])
            
            GluMeas = self.body.mmol_to_mgdl(state_vec[state_indices["GluInte"]])

            current_time = i * time_step
            events: Tuple[Bolus, Meal] = self.scenario.get_events(current_time)
            bolus, meal = events
            bg_raw = int(GluMeas)
            bg = self.cgm.get_reading(bg_raw)

                         
            # handle meal/bolus
            if bolus:
                (bolus_bg, bolus) = self.process_bolus(bolus, bg, state_vec)
                resume = self.scenario.user_config.resume if hasattr(self.scenario, 'user_config') else None
                self.pump.send_bolus_command(bolus_bg, bolus, resume)
            dose = self.pump.pump_emulator.step_minute(bg=bg)
            
            self.logger.step(dose, time_step)
            
            r = ode(lambda t, state: self.body.model(current_time + t, state, dose))
            r.set_initial_value(state_vec[:self.body.num_variables])
            res: np.ndarray = r.integrate(r.t + time_step)
            final = res.flatten()
            
            # body state
            state_vec[:self.body.num_variables] = final
            
            # self.apply_analyses(current_time, state_vec)
            
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = state_vec
        return trace
    
    def get_error_info(self) -> ErrorInfo:
        err_info = self.logger.error_dump()
        err_info.scenario = self.scenario
        return err_info