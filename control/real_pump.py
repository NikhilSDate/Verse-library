
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
    'Gs': 0,
    'X': 1,
    'Isc1': 2,
    'Isc2': 3,
    'Gt': 4,
    'Gp': 5,
    'Il': 6,
    'Ip': 7,
    'I1': 8,
    'Id': 9, 
    'pump_iob_1': 10,
    'pump_elapsed_1': 11,
    'pump_iob': 12
    }
    
    body_params = {
        'VG': 1.88 # distribution volume of glucose (dl/kg)
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
        Gs, X, Isc1, Isc2, Gt, Gp, Il, Ip, I1, Id = state[:10]

        um = 0 # um(t), glucose rate of appearance
        uI = 0 # uI(t) (maybe uU/min???)

        X_dot = -0.0278 * X + 0.0278 * (18.2129 * Ip - 100.25)
        Isc1_dot = 0.0142 * Isc1 - 0.0078 * Isc2 + uI
        Isc2_dot = 0.0152 * Isc1 - 0.0078 * Isc2
        Gt_dot = -0.0039 * (3.2267 + 0.0313 * X) * Gt * (1 - (0.0026 * Gt) + (2.5097 * pow(10, -6) * pow(Gt, 2))) + (0.0581 * Gp) - (0.0871 * Gt)

        Gp_dot = 3.7314 - (0.0047 * Gp) - (0.0121 * Id) - (0.0581 * Gp) + (0.0871 * Gt) + um
        Il_dot = -0.4219 * Il + 0.225 * Ip
        Ip_dot = -0.315 * Ip + 0.1545 * Il + 1.9 * pow(10, -3) * Isc1 + 7.8 * pow(10, -3) * Isc2
        I1_dot = -0.0046 * (I1 - 18.2129 * Ip)
        Id_dot = -0.0046 * (Id - I1)
        Gs_dot = 0.1 * (0.5521 * Gp - Gs)
        
        
        return [Gs_dot, X_dot, Isc1_dot, Isc2_dot, Gt_dot, Gp_dot, Il_dot, Ip_dot, I1_dot, Id_dot, 0, 0, 0]

    
    @staticmethod
    def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "real_pump.py")
        # will need to figure out how to incorporate carbs into this later
        # init[0][PumpAgent.state_indices['body_carbs']] = carbs_low
        # init[1][PumpAgent.state_indices['body_carbs']] = carbs_high
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

        glucose = PumpAgent.get_visible(init)[0] # THIS NEEDS TO BE CHANGED DEPENDING ON BODY MODEL
        # carbs = init[PumpAgent.state_indices['body_carbs']] # THIS NEEDS TO BE CHANGED DEPENDING ON BODY MODEL
        
        iob_1 = init[PumpAgent.state_indices['pump_iob_1']]
        elapsed_1 = init[PumpAgent.state_indices['pump_elapsed_1']]
        iob = init[PumpAgent.state_indices['pump_iob']]
        # keep insulin duration constant at 4 hrs for now
        iob_state = ([(iob_1, elapsed_1, 14400)], iob, 14400)
        pump = Pump(iob_state)
        print('glucose', glucose)
        dose = pump.dose_simple(glucose, 100) # hardcode carbs to 20 for now
        print('dose', dose)
        iob_array, iob, max_duration = pump.get_state()
        iob_1, elapsed_1 = iob_array[0][0], iob_array[0][1]
        init[PumpAgent.state_indices['pump_iob_1']] = iob_1
        init[PumpAgent.state_indices['pump_elapsed_1']] = elapsed_1
        init[PumpAgent.state_indices['pump_iob']] = iob
        for i in range(0, num_points):
            init[PumpAgent.state_indices['Isc1']] += dose * 1000 / 80
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
            dose = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace
    
    @staticmethod
    def get_init_state(init_bg, pump_state=[0, 0, 0]):
        body_state = [0] * 10
        body_state[PumpAgent.state_indices['Gp']] = init_bg * PumpAgent.body_params['VG']
        return body_state + pump_state
    
    @staticmethod
    def get_visible(state):
        glucose = state[PumpAgent.state_indices['Gp']] / PumpAgent.body_params['VG']
        return (glucose,)
        
class ThermoMode(Enum):
    A=auto()
    B=auto()

class State:
    # body model
    Gs: float # subcutanous glucose concentration ??? (is this plasma glucose concentration mg/dL)
    X: float # insulin conc. in remote chamber/insulin in the interstitial fluid: pmol/L
    Isc1: float # subcutaneous insulin in chamber 1 ???
    Isc2: float # subcutaneous insulin in chamber 2 ???
    Gt: float # Glcuose conc. in rapidly equilibriating tissues/ glucose MASS in rapidly equilibriating tissues: mg/kg
    Gp: float # Glucose conc. in plasma/ glucose MASS in plasma: mg/kg
    Il: float # Portal vein insulin mass/insulin mass in liver.: pmol/kg
    Ip: float # Insulin mass in plasma: pmol/kg
    I1: float # Insulin chamber #1 concentration: pmol/L
    Id: float # delayed insulin from chamber 1/ delayed insulin signal realized with a chain of two compartments: pmol/L
    
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
    init = [PumpAgent.get_init_state(120), PumpAgent.get_init_state(130)]
    result1, tree1 = PumpAgent.verify_bolus(init, 0, 0, duration=120)
    # result2, tree2 = PumpAgent.verify_bolus(result1, 10, 20, duration=120)
    # PumpAgent.link_nodes(tree1.root, tree2.root)
    # result2, tree2 = PumpAgent.verify_bolus(copy.deepcopy(result1), 0, 0, duration=60)
    fig = go.Figure() 
    fig = reachtube_tree(tree1, None, fig, 0, 6)
    fig.show()
    breakpoint()

