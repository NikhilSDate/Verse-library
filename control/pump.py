
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
import os 



class Pump:
    kCarbs = 200
    kGlucose = 150
    kTarget = 120
    kMaxBolus = 30

    def __init__(self, iob):
        self.iob = iob
    
    def dose(self, carbs, glucose):
        amount = int(carbs / 5) * 5 / Pump.kCarbs + (int(glucose / 5) * 5 - Pump.kTarget) / Pump.kGlucose
        amount = min(max(amount, 0), Pump.kMaxBolus)
        self.iob += amount
        return amount
    
    def iob_update(self):
        self.iob = np.exp(-0.01) * self.iob



class PumpAgent(BaseAgent):
   
    kI = 0.01
    kG = 0.05
    kIG = 1
    kC = 0.05

    def __init__(
        self, 
        id, 
        code = None,
        file_name = None
    ):
        super().__init__(id, code, file_name)


    @staticmethod
    def insulin_glucose_model(t, state):
        insulin, glucose = state[0], state[1]
        carbs = state[3]
        insulin_dot = -PumpAgent.kI * insulin
        glucose_dot = carbs * PumpAgent.kG - PumpAgent.kIG * insulin
        carbs_dot = -PumpAgent.kC * carbs
        return [insulin_dot, glucose_dot, 0, carbs_dot]

    
    @staticmethod
    def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "pump.py")
        init[0][3] = carbs_low
        init[1][3] = carbs_high
        agent = PumpAgent('pump', file_name=input_code_name)
        scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

        scenario.add_agent(agent) ### need to add breakpoint around here to check decision_logic of agents
        # -----------------------------------------

        scenario.set_init_single(
            'pump', init, (ThermoMode.A,)
        )

        # assumption: meal every 12 hours

        traces = scenario.verify(duration, time_step)
        end_low = traces.root.trace['pump'][-2]
        end_high = traces.root.trace['pump'][-1]
        return [end_low[1:], end_high[1:]], traces
    
    @staticmethod
    def link_nodes(node1, node2):
        agent = list(node1.trace.keys())[0]
        trace1_end = node1.trace[agent][-1][0]
        trace2_len = len(node2.trace[agent])
        for i in range(trace2_len):
            node2.trace[agent][i][0] += trace1_end
        node2.height = node1.height + 1
        node2.id = node1.id + 1
        node1.child.append(node2)

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        glucose = init[1]
        iob = init[2]
        carbs = init[3]
        pump = Pump(iob)
        dose = pump.dose(carbs, glucose)
        for i in range(0, num_points):
            init = [init[0] + dose, init[1], pump.iob, init[3]]
            r = ode(lambda t, state: self.insulin_glucose_model(t, state))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            pump.iob_update()
            init[2] = pump.iob
            dose = 0.1 / 60
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    A=auto()
    B=auto()

class State:
    insulin: float
    glucose: float
    iob: float
    carbs: float
    agent_mode: ThermoMode 

    def __init__(self, x, agent_mode: ThermoMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output 



if __name__ == "__main__":
    init = [[0, 120, 0, 50], [0, 120, 0, 50]]
    result1, tree1 = PumpAgent.verify_bolus(init, 50, 50, duration=60)
    result2, tree2 = PumpAgent.verify_bolus(copy.deepcopy(result1), 0, 0, duration=60)
    PumpAgent.link_nodes(tree1.root, tree2.root)
    result3, tree3 = PumpAgent.verify_bolus(copy.deepcopy(result2), 0, 0, duration=60)
    PumpAgent.link_nodes(tree2.root, tree3.root)
    result4, tree4 = PumpAgent.verify_bolus(copy.deepcopy(result3), 100, 150, duration=60)
    PumpAgent.link_nodes(tree3.root, tree4.root)
    result5, tree5 = PumpAgent.verify_bolus(copy.deepcopy(result4), 0, 0, duration=60)
    PumpAgent.link_nodes(tree4.root, tree5.root)
    result6, tree6 = PumpAgent.verify_bolus(copy.deepcopy(result5), 0, 0, duration=60)
    PumpAgent.link_nodes(tree5.root, tree6.root)
    fig = go.Figure() 
    fig = reachtube_tree(tree1, None, fig, 0, 2)
    fig.show()
    breakpoint()

