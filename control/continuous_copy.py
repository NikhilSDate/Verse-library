
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
    def dynamic_heat(t, state, heater):
        T = state[0]
        T_dot = heater * ThermoAgent.heat_gain_rate + (ThermoAgent.ambient - T) * ThermoAgent.heat_loss_rate
        return [T_dot, 0]
    
    @staticmethod
    def dynamic_cool(t, state):
        T = state[0]
        T_dot = 0 * ThermoAgent.heat_gain_rate + (ThermoAgent.ambient - T) * ThermoAgent.heat_loss_rate
        return [T_dot, 0]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            heater = 0.1 * init[1]
            r = ode(lambda t, state: self.dynamic_heat(t, state, heater))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            init[1] += 0.1 * (90 - init[0])
            init[1] = min(max(init[1], -10), 10)
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    A=auto()
    B=auto()

class State:
    x: float
    error: float
    agent_mode: ThermoMode 

    def __init__(self, x, agent_mode: ThermoMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "continuous_copy.py")
    Thermo = ThermoAgent('thermo', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(Thermo) ### need to add breakpoint around here to check decision_logic of agents

    init_bruss = [[20, 0], [120, 0]] # setting initial upper bound to 72 causes hyperrectangle to become large fairly quickly
    # -----------------------------------------

    scenario.set_init_single(
        'thermo', init_bruss, (ThermoMode.A,)
    )

    # basis = np.array([1]) * np.diag([0.1]) # this doesn't actually make sense, but not sure how algorithm actually handles 1d polytopes
    # center = np.array([68.5])
    # C = np.transpose(np.array([1,-1]))
    # g = np.array([1,1])


    # Thermo.set_initial(
    #     StarSet(center, basis, C, g)
    #     , (ThermoMode.Heat,)
    # )

    # scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    # scenario.set_sensor(BaseStarSensor())
    ### t=10 takes quite a long time to run, try t=4 like in c2e2 example
    ### seems to actually loop at t=4.14, not sure what that is about -- from first glance, reason seems to be hyperrectangles blowing up in size
    traces = scenario.verify(200, 0.5)
    fig = go.Figure() 
    fig = reachtube_tree(traces, None, fig, 0, 2)
    fig.show()

