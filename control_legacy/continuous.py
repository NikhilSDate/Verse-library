
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

class ThermoAgent(BaseAgent):
    # how quickly heat is lost to the ambient environment
    heat_loss_rate = 0.015
    # how quickly heat is gained with the heater
    # this ratio is desired for similar behavior as the physical model
    heat_gain_rate = heat_loss_rate * 1000

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
    def dynamic_heat(t, state):
        T = state[0]
        heater = state[1]
        T_dot = heater * ThermoAgent.heat_gain_rate + (ThermoAgent.ambient - T) * ThermoAgent.heat_loss_rate
        return [T_dot, 0, 1]
    
    @staticmethod
    def dynamic_cool(t, state):
        T = state[0]
        T_dot = 0 * ThermoAgent.heat_gain_rate + (ThermoAgent.ambient - T) * ThermoAgent.heat_loss_rate
        return [T_dot, 0, 1]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            r = ode(self.dynamic_heat)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    A=auto()
    B=auto()

class State:
    x: float
    heater_output: float
    wait_time: float
    agent_mode: ThermoMode 

    def __init__(self, x, agent_mode: ThermoMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    
    if ego.agent_mode == ThermoMode.A and ego.x <= 70:
        output.agent_mode = ThermoMode.B
        output.heater_output = 1

    if ego.agent_mode == ThermoMode.A and 70 < ego.x <= 80:
        output.agent_mode = ThermoMode.B
        output.heater_output = 0.1* (80 - ego.x)

    if ego.agent_mode == ThermoMode.A and ego.x > 80:
        output.agent_mode = ThermoMode.B
        output.heater_output = 0
    
    if ego.agent_mode == ThermoMode.B and ego.x <= 70:
        output.agent_mode = ThermoMode.A
        output.heater_output = 1

    if ego.agent_mode == ThermoMode.B and 70 < ego.x <= 80:
        output.agent_mode = ThermoMode.A
        output.heater_output = 0.1* (80 - ego.x)

    if ego.agent_mode == ThermoMode.B and ego.x > 80:
        output.agent_mode = ThermoMode.A
        output.heater_output = 0
    
    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "continuous.py")
    Thermo = ThermoAgent('thermo', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(Thermo) ### need to add breakpoint around here to check decision_logic of agents

    init_bruss = [[70, 0, 0], [70, 0, 0]] # setting initial upper bound to 72 causes hyperrectangle to become large fairly quickly
    # -----------------------------------------

    scenario.set_init_single(
        'thermo', init_bruss, (ThermoMode.A,)
    )

    traces = scenario.verify(25, 0.1)
    fig = go.Figure() 
    fig = reachtube_tree(traces, None, fig, 0, 1)
    fig.show()

