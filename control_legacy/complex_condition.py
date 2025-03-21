
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
    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)

    @staticmethod
    def dynamic_heat(t, state):
        return [1]
    
    @staticmethod
    def dynamic_cool(t, state):
        return [-2]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            if mode[0]=="Heat":
                r = ode(self.dynamic_heat)
            elif mode[0]=="Cool":
                r = ode(self.dynamic_cool)
            else:
                raise ValueError
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    Heat=auto()
    Cool=auto()

class State:
    x: float
    agent_mode: ThermoMode 

    def __init__(self, x, agent_mode: ThermoMode):
        pass 
    
def complicated_transform(x):
    return (x - 2)*(x - 5)*(x - 10)*(x - 15)

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)

    if ego.agent_mode == ThermoMode.Heat and complicated_transform(ego.x) < 0:
        output.agent_mode = ThermoMode.Cool
    if ego.agent_mode == ThermoMode.Cool and complicated_transform(ego.x) >= 0:
        output.agent_mode = ThermoMode.Heat

    return output 



if __name__ == "__main__":
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "complex_condition.py")
    Thermo = ThermoAgent('thermo', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(Thermo) ### need to add breakpoint around here to check decision_logic of agents

    init_bruss = [[-5], [-1]] # setting initial upper bound to 72 causes hyperrectangle to become large fairly quickly
    # -----------------------------------------

    scenario.set_init_single(
        'thermo', init_bruss, (ThermoMode.Heat,)
    )

    ### t=10 takes quite a long time to run, try t=4 like in c2e2 example
    ### seems to actually loop at t=4.14, not sure what that is about -- from first glance, reason seems to be hyperrectangles blowing up in size
    trace = scenario.verify(10, 0.5)
    fig = go.Figure()
    fig = reachtube_tree(trace, None, fig, 0, 1, [0, 1], "lines", "trace")
    fig.show()