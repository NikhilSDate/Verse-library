
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
from real_pump_model import *
from stateutils import set, get, state_indices, num_continuous_variables # work toward removing the state_indices import

sys.path.insert(1, '/home/ndate/Research/insulin_pump/unicorn_analyzer')

from pump_wrapper import Pump


class PumpAgent(BaseAgent):
    
    
            
    body_params = {
        'VG': 1.88, # distribution volume of glucose (dl/kg)
        'weight': 70 # weight of person in kg
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
    def units_to_pmol_per_kg(units):
        # use factor 1 milliunit = 6 pmol
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501531/
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769591/pdf/dst-01-0323.pdf says insulin infusion rate is pmol/kg/min
        return units * 1000 * 6 / PumpAgent.body_params['weight']
    
    @staticmethod
    def extract_pump_state(init, pump, events=3):
        iob_array, iob, max_duration = pump.get_state()
        set(init, 'pump_iob', iob)
        for i in range(events):
            set(init, f'pump_iob_{i}', iob_array[i][0])
            set(init, f'pump_elapsed_{i}', iob_array[i][1])

    @staticmethod
    def get_initialized_pump(init, events=3, duration=4):
        # assume same duration for every item
        iob_items = []
        for i in range(events):
            item_iob = get(init, f'pump_iob_{i}')
            item_elapsed = get(init, f'pump_elapsed_{i}')
            iob_items.append((item_iob, item_elapsed, duration))
        iob = get(init, 'pump_iob')
        pump = Pump((iob_items, iob, duration))
        return pump

    @staticmethod
    def insulin_glucose_model(t, state):
        Gs, X, Isc1, Isc2, Gt, Gp, Il, Ip, I1, Id = state[:10]

        um = 0 # um(t), glucose rate of appearance
        uI = 0 # uI(t) (maybe pmol/kg/min???)


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
        
        body_derivatives = [Gs_dot, X_dot, Isc1_dot, Isc2_dot, Gt_dot, Gp_dot, Il_dot, Ip_dot, I1_dot, Id_dot]
        pump_derivatives = [0 for i in range(num_continuous_variables - len(body_derivatives))]
        return body_derivatives + pump_derivatives

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ) -> TraceType:
        print('tracing', init, time_bound, time_step)
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        glucose = PumpAgent.get_visible(init)[0] # THIS NEEDS TO BE CHANGED DEPENDING ON BODY MODEL
        # carbs = init[state_indices['body_carbs']] # THIS NEEDS TO BE CHANGED DEPENDING ON BODY MODEL
        
        # keep insulin duration constant at 4 hrs for now
        
        pump = PumpAgent.get_initialized_pump(init)
        PumpAgent.extract_pump_state(init, pump)
        dose = 0
        print('num_points', num_points)
        for i in range(0, num_points):
            init[state_indices['Isc1']] += PumpAgent.units_to_pmol_per_kg(dose)
            r = ode(lambda t, state: self.insulin_glucose_model(t, state))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            if i * time_step in [0]:
                glucose = PumpAgent.get_visible(init)[0]
                print('glucose', glucose)
                dose = pump.dose_simple(glucose, 0)
                print('dose', dose)
            else:
                dose = 0
            PumpAgent.extract_pump_state(init, pump)
            pump.delay()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace
    
    @staticmethod
    def get_init_state(init_bg, pump_state=[0, 0, 0]):
        state = [0 for i in range(num_continuous_variables)]
        state[state_indices['Gp']] = init_bg * PumpAgent.body_params['VG']
        # set(state, 'Gt', 141)
        # set(state, 'Ip', 5.5043)
        return state
    
    @staticmethod
    def get_visible(state):
        glucose = state[state_indices['Gp']] / PumpAgent.body_params['VG']
        return (glucose,)

def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "real_pump_model.py")
    # will need to figure out how to incorporate carbs into this later
    # init[0][state_indices['body_carbs']] = carbs_low
    # init[1][state_indices['body_carbs']] = carbs_high
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

def simulate_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "real_pump_model.py")
    # will need to figure out how to incorporate carbs into this later
    # init[0][state_indices['body_carbs']] = carbs_low
    # init[1][state_indices['body_carbs']] = carbs_high
    agent = PumpAgent('pump', file_name=input_code_name)
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(agent) ### need to add breakpoint around here to check decision_logic of agents
    # -----------------------------------------

    scenario.set_init_single(
        'pump', init, (ThermoMode.A,)
    )

    # assumption: meal every 12 hours

    traces = scenario.simulate(duration, time_step)
    end = traces.root.trace['pump'][-1]
    return [end[1:], end[1:]], traces

def link_nodes(node1, node2):
    agent = list(node1.trace.keys())[0]
    trace1_end = node1.trace[agent][-1][0]
    trace2_len = len(node2.trace[agent])
    for i in range(trace2_len):
        node2.trace[agent][i][0] += trace1_end
    node2.height = node1.height + 1
    node2.id = node1.id + 1
    node1.child.append(node2)

# boluses is time, [carbs_low, carbs_high]
def verify_boluses(init, boluses, duration=120):
    result1, tree1 = verify_bolus(init, boluses[0][1][0], boluses[0][1][1], duration)
    prev_result, prev_tree = result1, tree1
    for i in range(1, len(boluses)):
        result, tree = verify_bolus(copy.deepcopy(prev_result), boluses[i][1][0], boluses[i][1][1], duration)
        link_nodes(prev_tree.root, tree.root)
        prev_result, prev_tree = result, tree
    return result1, tree1

def simulate_boluses(init, boluses, duration=120):
    result1, tree1 = simulate_bolus(init, boluses[0][1], boluses[0][1], duration)
    prev_result, prev_tree = result1, tree1
    for i in range(1, len(boluses)):
        result, tree = simulate_bolus(copy.deepcopy(prev_result), boluses[i][1], boluses[i][1], duration)
        link_nodes(prev_tree.root, tree.root)
        prev_result, prev_tree = result, tree
    return result1, tree1

def plot_variable(fig, tree, var, mode: Union['simulate', 'verify']='verify'):
    idx = state_indices[var] + 1 # time is 0, so 1-index
    if mode == 'verify':
        fig = reachtube_tree(tree, None, fig, 0, idx)
    else:
        fig = simulation_tree(tree, None, fig, 0, idx)
    fig.show()

if __name__ == "__main__":
    init = [PumpAgent.get_init_state(140), PumpAgent.get_init_state(150)]
    result, tree = verify_boluses(init, [[0, [50, 70]], [0, [50, 70]], [0, [50, 70]], [0, [50, 70]]], duration=120)
    # result, tree = simulate_boluses(init, [[0, 50], [120, 10]])
    fig = go.Figure()
    plot_variable(fig, tree, 'Gp', mode='verify')
    breakpoint()

