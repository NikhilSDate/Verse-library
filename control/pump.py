
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

class Pump:
    kCarbs = 150
    kGlucose = 150
    kTarget = 120
    kMaxBolus = 30

    def __init__(self, iob):
        self.iob = iob
    
    def dose(self, carbs, glucose):
        amount = carbs / Pump.kCarbs + (glucose - Pump.kTarget) / Pump.kGlucose
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

    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        insulin = init[0]
        glucose = init[1]
        iob = init[2]
        carbs = init[3]
        pump = Pump(iob)
        dose = pump.dose(carbs, glucose)
        init = [insulin + dose, glucose, pump.iob, carbs]
        trace[1, 0] = time_step
        trace[1, 1:] = init
        for i in range(1, num_points):
            r = ode(lambda t, state: self.insulin_glucose_model(t, state))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            pump.iob_update()
            init[2] = pump.iob
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
    import os 
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "pump.py")
    Thermo = PumpAgent('pump', file_name=input_code_name)

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(Thermo) ### need to add breakpoint around here to check decision_logic of agents

    init_bruss = [[0, 200, 0, 5], [0, 300, 0, 5]] # setting initial upper bound to 72 causes hyperrectangle to become large fairly quickly
    # -----------------------------------------

    scenario.set_init_single(
        'pump', init_bruss, (ThermoMode.A,)
    )

    # assumption: meal every 12 hours

    traces = scenario.verify(360, 1)

    end_low = traces.root.trace['pump'][-2]
    end_high = traces.root.trace['pump'][-1]
    breakpoint()
    fig = go.Figure() 
    fig = reachtube_tree(traces, None, fig, 0, 2)
    fig.show()

