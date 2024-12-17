from typing import Tuple, List 

import numpy as np 
from scipy.integrate import ode 

from verse import BaseAgent, Scenario, ScenarioConfig
from verse.analysis.analysis_tree import TraceType, AnalysisTree 
from verse.parser import ControllerIR
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType
import copy 

from enum import Enum, auto

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go

from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *
from verse.analysis.verifier import ReachabilityMethod
import sys

sys.path.append('/root/insulin_pump')

from thermostat_new import Thermostat


class ThermoAgent(BaseAgent):

    # how quickly heat is lost to the ambient environment
    heat_loss_rate = 0.015
    # how quickly heat is gained with the heater
    # this ratio is desired for similar behavior as the physical model
    heat_gain_rate = heat_loss_rate * 100

    ambient = 70
    temp = 90

    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic_heat(t, state, heater, mode):
        T = state[0]
        T_dot = heater * ThermoAgent.heat_gain_rate + (ThermoAgent.ambient - T) * ThermoAgent.heat_loss_rate

        OutputSum_1_64 = state[1]
        LastInput_2_64 = state[2]
        raw_temp_3_32 = (T + 22.2)/0.175
        if ThermoMode.State1.name in mode:
          OutputSum = 1.0
          LastInput = ((raw_temp_3_32 * 0.175) + -22.2)
        elif ThermoMode.State2.name in mode:
          OutputSum = 0.0
          LastInput = ((raw_temp_3_32 * 0.175) + -22.2)
        elif ThermoMode.State3.name in mode:
          OutputSum = ((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0))
          LastInput = ((raw_temp_3_32 * 0.175) + -22.2)
        else:
          raise RuntimeError(f'unknown mode {mode}')

        return [T_dot, OutputSum, LastInput, raw_temp_3_32]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        print('simulating', init, time_bound, time_step)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        thermo = Thermostat(firmware_path='files/thermostat-nonmanual.bin', addrs_path='files/addrs-nonmanual.yaml')
        for i in range(num_points):
            # control output goes here
            temp = init[0]
            control_output = thermo.update(temp, time_step)
            r = ode(lambda t, state: self.dynamic_heat(t, state, control_output, mode))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    State1 = auto()
    State2 = auto()
    State3 = auto()

class FlipMode(Enum):
    F1 = auto()
    F2 = auto()

class State:
    T: float
    OutputSum: float
    LastInput: float
    raw_temp: float
    agent_mode: ThermoMode 

    def __init__(self, x, time, agent_mode: ThermoMode):
        pass 

def decisionLogic(ego: State):
    state = copy.deepcopy(ego)

    OutputSum_1_64 = state.OutputSum
    LastInput_2_64 = state.LastInput
    raw_temp_3_32 = state.raw_temp
    if (ego.agent_mode == ThermoMode.State2 or ego.agent_mode == ThermoMode.State3) and (((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) > 1.0) and (not (1.0 < ((1.0 - ((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 0.0)) + 0.0))) and (not (0.0 > ((1.0 - ((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 0.0)) + 0.0))):
      state.agent_mode = ThermoMode.State1
    elif (ego.agent_mode == ThermoMode.State1 or ego.agent_mode == ThermoMode.State3) and (not (((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) > 1.0)) and (((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) < 0.0) and (not (1.0 < ((0.0 - ((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 0.0)) + 0.0))) and (not (0.0 > ((0.0 - ((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 0.0)) + 0.0))):
      state.agent_mode = ThermoMode.State2
    elif (ego.agent_mode == ThermoMode.State1 or ego.agent_mode == ThermoMode.State2) and (not (((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) > 1.0)) and (not (((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) < 0.0)) and (not (1.0 < ((((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) - ((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 0.0)) + 0.0))) and (not (0.0 > ((((((95.0 + -((raw_temp_3_32 * 0.175) + -22.2)) * 80.0) + OutputSum_1_64) + -((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 2.0)) - ((((raw_temp_3_32 * 0.175) + -22.2) + -LastInput_2_64) * 0.0)) + 0.0))):
      state.agent_mode = ThermoMode.State3

    return state


if __name__ == "__main__":

    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "thermostat.py")
    Thermo = ThermoAgent('thermo', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(parallel=False))

    scenario.add_agent(Thermo) ### need to add breakpoint around here to check decision_logic of agents

    init_state = [[60, 0, 0, 0], [80, 0, 0, 0]] # setting initial upper bound to 72 causes hyperrectangle to become large fairly quickly
    # -----------------------------------------

    scenario.set_init(
        [init_state], [(ThermoMode.State1,)]
    )
    traces_veri = scenario.verify(60, 0.1)
    # traces_simu = scenario.simulate_simple(180, 0.1)
    fig = go.Figure() 
    fig = reachtube_tree(traces_veri, None, fig, 2, 1)
    # fig = simulation_tree(traces_simu, None, fig, 2, 1)
    fig.write_image('out.png')

