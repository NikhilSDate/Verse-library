
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
import sys 

sys.path.insert(1, '/home/ndate/Research/insulin_pump/unicorn_analyzer')

from pump_wrapper import Pump

class PumpAgent(BaseAgent):
    
    state_indices = {
    'body_insulin': 0,
    'body_glucose': 1,
    'body_carbs': 2,
    'pump_iob_1': 3,
    'pump_elapsed_1': 4,
    'pump_iob': 5
    }

    kI = 0.01
    kG = 0.05
    kIG = 0.1
    kC = 0.07

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
        carbs = state[2]
        insulin_dot = -PumpAgent.kI * insulin
        glucose_dot = carbs * PumpAgent.kG - PumpAgent.kIG * insulin
        carbs_dot = -PumpAgent.kC * carbs
        return [insulin_dot, glucose_dot, carbs_dot, 0, 0, 0]

    
    @staticmethod
    def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "real_pump.py")
        init[0][PumpAgent.state_indices['body_carbs']] = carbs_low
        init[1][PumpAgent.state_indices['body_carbs']] = carbs_high
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
        print('tracing', init, time_bound)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        glucose = init[PumpAgent.state_indices['body_glucose']] # THIS NEEDS TO BE CHANGED DEPENDING ON BODY MODEL
        carbs = init[PumpAgent.state_indices['body_carbs']] # THIS NEEDS TO BE CHANGED DEPENDING ON BODY MODEL
        
        iob_1 = init[PumpAgent.state_indices['pump_iob_1']]
        elapsed_1 = init[PumpAgent.state_indices['pump_elapsed_1']]
        iob = init[PumpAgent.state_indices['pump_iob']]
        # keep insulin duration constant at 4 hrs for now
        iob_state = ([(iob_1, elapsed_1, 14400)], iob, 14400)
        pump = Pump(iob_state)
        dose = pump.dose_simple(glucose, carbs)
        iob_array, iob, max_duration = pump.get_state()
        iob_1, elapsed_1 = iob_array[0][0], iob_array[0][1]
        init[PumpAgent.state_indices['pump_iob_1']] = iob_1
        init[PumpAgent.state_indices['pump_elapsed_1']] = elapsed_1
        init[PumpAgent.state_indices['pump_iob']] = iob
        for i in range(0, num_points):
            init[PumpAgent.state_indices['body_insulin']] += dose
            r = ode(lambda t, state: self.insulin_glucose_model(t, state))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            pump.delay()
            iob_array, iob, max_duration = pump.get_state()
            iob_1, elapsed_1 = iob_array[0][0], iob_array[0][1]
            init[PumpAgent.state_indices['pump_iob_1']] = iob_1
            init[PumpAgent.state_indices['pump_elapsed_1']] = elapsed_1
            init[PumpAgent.state_indices['pump_iob']] = iob
            dose = 0.5 / 60
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

class ThermoMode(Enum):
    A=auto()
    B=auto()

class State:
    # body model
    body_insulin: float
    body_glucose: float
    body_carbs: float
    
    # pump model
    pump_iob_1: float
    pump_elapsed_1: int
    pump_iob: float

    agent_mode: ThermoMode 

    def __init__(self, x, agent_mode: ThermoMode):
        pass 

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output 



if __name__ == "__main__":
    init = [[0, 125, 0, 0, 0, 0], [0, 130, 0, 0, 0, 0]]
    result1, tree1 = PumpAgent.verify_bolus(init, 0, 0, duration=120)
    result2, tree2 = PumpAgent.verify_bolus(result1, 10, 20, duration=120)
    PumpAgent.link_nodes(tree1.root, tree2.root)
    # result2, tree2 = PumpAgent.verify_bolus(copy.deepcopy(result1), 0, 0, duration=60)
    fig = go.Figure() 
    fig = reachtube_tree(tree1, None, fig, 0, 6)
    fig.show()
    breakpoint()

