from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
from math import tanh

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
from real_pump_matlab_model import *
from matlab_body_model import model, basal_states, Vg
from matlab_stateutils import (
    set,
    get,
    state_indices,
    num_continuous_variables,
    num_meals,
    state_variable_names
)  # work toward removing the state_indices import

import pandas as pd
from tqdm import tqdm

from scenario import SimulationScenario, Bolus, BolusType

import pickle



sys.path.insert(1, "/home/ndate/Research/insulin_pump/unicorn_analyzer")

from pump_wrapper import Pump

'''
SCENARIO: PUMP'S TARGET BG NOT EQUAL TO BODY'S BASAL BG
'''

class PumpAgent(BaseAgent):

    body_params = {
        "BW": 78,  # weight of person in kg
        "Gb": 110 # match pump's target BG with basal BG
    }
    


    def __init__(self, id, simulation_scenario, code=None, file_name=None):
        super().__init__(id, code, file_name)
        self.scenario:SimulationScenario = simulation_scenario

    def TC_simulate(self, mode: List[str], init, time_bound, time_step, lane_map=None) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init

        # keep insulin duration constant at 4 hrs for now
        
        pump = get_initialized_pump(init)
        meals = get_meals_from_state(init)
        dose = 0
        for i in tqdm(range(0, num_points)):
            current_time = i * time_step
            init[state_indices["Isc1"]] += units_to_pmol_per_kg(dose)
            r = ode(lambda t, state: insulin_glucose_model(current_time + t, state, meals))
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            init[0] = get(init, 'Gp') / Vg
            bolus = self.scenario.get_bolus(current_time)
            if bolus:                
                dose = handle_bolus(pump, init, bolus)
                tqdm.write(str(dose))
            elif current_time % 5 == 0:
                # TODO: need to actually model basal deliveries
                dose = self.scenario.basal_rate / 60  * 5 # basal rate = 0.5u/hr
            else:
                dose = 0
            pump.delay()
            extract_pump_state(init, pump)
            # pump.delay()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

def units_to_pmol_per_kg(units):
    # use factor 1 milliunit = 6 pmol
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501531/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769591/pdf/dst-01-0323.pdf says insulin infusion rate is pmol/kg/min
    
    # but it looks like the matlab model uses 1 milliunit = 6.9444 pmol
    
    return units * 6944.4 / PumpAgent.body_params["BW"]

def extract_pump_state(init, pump, events=3):
    iob_array, iob, max_duration = pump.get_state()
    set(init, "pump_iob", iob)
    for i in range(events):
        set(init, f"pump_iob_{i}", iob_array[i][0])
        set(init, f"pump_elapsed_{i}", iob_array[i][1])

def get_initialized_pump(init, events=3, duration=4):
    # assume same duration for every item
    iob_items = []
    for i in range(events):
        item_iob = get(init, f"pump_iob_{i}")
        item_elapsed = get(init, f"pump_elapsed_{i}")
        iob_items.append((item_iob, item_elapsed, duration))
    iob = get(init, "pump_iob")
    pump = Pump((iob_items, iob, duration))
    return pump

def insulin_glucose_model(t, state, meals):
    body_vars = state[:18] # have 17 variables plus one ignore variable
    body_derivatives = model(t, body_vars, '', PumpAgent.body_params['BW'], PumpAgent.body_params['Gb'], 
                                meals, 0, 0, 0)

    pump_derivatives = np.zeros((num_continuous_variables - len(body_derivatives),))
    return np.concatenate([body_derivatives, pump_derivatives])

def get_meals_from_state(state):
    result = []
    for i in range(1, num_meals + 1):
        Di = get(state, f'D_{i}')
        ti = get(state, f't_{i}')
        result.append((Di, ti))
    return result

def get_init_state(init_bg, meals):
    (Gb,Gpb,Gtb,Ilb,Ipb,Ipob,Ib,IIRb,Isc1ss,Isc2ss,kp1,Km0,Hb,SRHb,Gth,SRsHb, XHb,Ith,IGRb,Hsc1ss,Hsc2ss) = basal_states(init_bg)
    body_init_state = [init_bg, Gpb,Gtb,Ilb,Ipb,Ib,Ib,0,0,0,0,SRsHb,Hb,XHb,Isc1ss,Isc2ss,Hsc1ss,Hsc2ss]
    scenario_state = []
    for i in range(len(meals)):
        carbs = meals[i][0] * 1000 # convert g to mg
        scenario_state.append(carbs) 
        scenario_state.append(meals[i][1])
    print(scenario_state)
    pump_init_state = [0 for i in range(num_continuous_variables - len(body_init_state) - len(scenario_state))] # exclude D
    return body_init_state + scenario_state + pump_init_state

# TODO should this be in the utils file instead?
def print_state(state):
    a = list(state_indices.keys())[:14]
    for var in a:
        print(f"\t{var}: {state[state_indices[var]]}")

def get_bg(state):
    glucose = state[state_indices["Gp"]] / Vg
    return glucose


def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "real_pump_model.py")
    agent = PumpAgent("pump", file_name=input_code_name)
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(
        agent
    )

    scenario.set_init_single("pump", init, (ThermoMode.A,))
    traces = scenario.verify(duration, time_step)
    end_low = traces.root.trace["pump"][-2]
    end_high = traces.root.trace["pump"][-1]
    return [end_low[1:], end_high[1:]], traces

def simulate_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "real_pump_model.py")
    agent = PumpAgent('pump', file_name=input_code_name)
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(agent)

    scenario.set_init_single(
        'pump', init, (ThermoMode.A,)
    )
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

def simulate_boluses(init, scenario):
    result1, tree1 = simulate_bolus(init, boluses[0][1], boluses[0][1], duration)
    prev_result, prev_tree = result1, tree1
    for i in range(1, len(boluses)):
        result, tree = simulate_bolus(copy.deepcopy(prev_result), boluses[i][1], boluses[i][1], duration)
        link_nodes(prev_tree.root, tree.root)
        prev_result, prev_tree = result, tree
    return result1, tree1

def handle_bolus(pump, state, bolus: Bolus) -> float:
    if bolus.type == BolusType.Simple:
        bg = get_bg(state)
        dose = pump.dose_simple(bg + 30, bolus.carbs)
        return dose
    else:
        # glucose = get_visible(init)[0]
        # carbs = time_to_carbs[current_time]
        # pump.dose_extended(glucose, carbs, 50, 120)
        # print('dosing')
        raise NotImplementedError()

def plot_variable(fig, tree, var, mode: Union['simulate', 'verify']='simulate', show=True):
    idx = state_indices[var] + 1 # time is 0, so 1-index
    if mode == 'verify':
        fig = reachtube_tree(tree, None, fig, 0, idx)
    else:
        fig = simulation_tree(tree, None, fig, 0, idx)
    if show:
        fig.show()
    return fig
    
def simulate_three_meal_scenario(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs):
    # start simulation at 8 AM
    meals = [(breakfast_carbs, 0), (lunch_carbs, 240), (dinner_carbs, 660)]
    init_state = get_init_state(init_bg, meals)
    init = [init_state, init_state]
    simulation_scenario = SimulationScenario(meals, basal_rate)
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "real_pump_matlab_model.py")
    agent = PumpAgent("pump", simulation_scenario=simulation_scenario, file_name=input_code_name)
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (ThermoMode.A,))
    duration = simulation_scenario.simulation_duration
    time_step = 1
    traces = scenario.simulate(duration, time_step)
    return traces

def generate_all_three_meal_traces(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs, trace_directory='traces/'):
    
    all_combinations = np.array(np.meshgrid(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs)).T.reshape(-1, 5)
    existing_files = os.listdir(trace_directory)
    for i in tqdm(range(len(all_combinations))):
        combination = all_combinations[i]
        tqdm.write(f'Simulating with init state {combination}')
        bg, br, bc, lc, dc = combination
        filename = f'trace_{bg}_{br}_{bc}_{lc}_{dc}.csv'
        if filename in existing_files:
            continue
        traces = simulate_three_meal_scenario(bg, br, bc, lc, dc)
        save_traces(traces, os.path.join(trace_directory, filename))
        
def plot_trace(filename, variable, trace_directory='traces/'):
    path = os.path.join(trace_directory, filename)
    df = pd.read_csv(path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['t'], y=df[variable]))
    fig.update_layout(dict(xaxis_title='t', yaxis_title=variable))
    fig.show()        

def verify_three_meal_scenario(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs):
    meals_low = [(breakfast_carbs[0], 0), (lunch_carbs[0], 240), (dinner_carbs[0], 660)]
    meals_high = [(breakfast_carbs[1], 0), (lunch_carbs[1], 240), (dinner_carbs[1], 660)]
    init_state_low = get_init_state(init_bg[0], meals_low)
    init_state_high = get_init_state(init_bg[1], meals_high)
    init = [init_state_low, init_state_high]
    simulation_scenario = SimulationScenario(meals_low, basal_rate)
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "real_pump_matlab_model.py")
    agent = PumpAgent("pump", simulation_scenario=simulation_scenario, file_name=input_code_name)
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (ThermoMode.A,))
    duration = simulation_scenario.simulation_duration
    time_step = 1
    traces = scenario.verify(duration, time_step)
    return traces


def save_traces(traces: AnalysisTree, filename):
    data = np.array(list(traces.root.trace.values())[0]) # we only have one agent
    cols = ['t'] + state_variable_names[:-1] # drop the discrete state
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filename)
    print(data.shape)    

if __name__ == "__main__":
    # traces = verify_three_meal_scenario([100, 130], 0, [50, 50], [100, 100], [100, 100])
    # breakpoint()
    # # fig = plot_variable(go.Figure(), traces, 'G', 'verify', show=False)
    # # fig.write_image('three_meals_verify.png')
    # # fig.show()
    # with open('three_meal_traces.pickle', 'wb') as f:
    #     pickle.dump(traces, f)
    # basal_rate = [0]
    # breakfast_carbs = [30, 50, 70]
    # lunch_carbs = [60, 80, 100]
    # dinner_carbs = [60, 80, 100]
    # # traces = generate_all_three_meal_traces(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs)
    # traces = simulate_three_meal_scenario(120, 0, 60, 100, 100)
    # plot_variable(go.Figure(), traces, 'G', 'simulate')
    # breakpoint()
    # plot_trace('trace_100_0_30_60_60.csv', 'Qgut')
    plot_trace('trace_100_0_30_60_60.csv', 'G')