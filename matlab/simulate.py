# import copy
# import os
import sys

# import pickle
from typing import *

# from enum import Enum, auto
# from math import tanh

# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from scipy.integrate import ode

# import plotly.graph_objects as go
from verse import BaseAgent, Scenario, ScenarioConfig

# from verse.analysis.analysis_tree import TraceType, AnalysisTree
# from verse.parser import ControllerIR
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType

# from verse.plotter.plotter2D import *
# from verse.plotter.plotter3D_new import *

# from verse.stars.starset import *
# from verse.sensor.base_sensor_stars import *
# from verse.analysis.verifier import ReachabilityMethod

# from pump_wrapper import Pump

# from body_model import model, basal_states, Vg
from verse_model import *

# from scenario import SimulationScenario, Bolus, BolusType
# from utils import (
#    set_val,
#    get_val,
#    state_indices,
#    num_continuous_variables,
#    num_meals,
#    state_variable_names,
# )  # work toward removing the state_indices import
from pump_agent import *

# sys.path.insert(1, os.environ["PUMP_PATH"])


"""
SCENARIO: PUMP'S TARGET BG NOT EQUAL TO BODY'S BASAL BG
"""


def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):
    script_dir = os.path.realpath(os.path.dirname(__file__))
    agent = PumpAgent("pump", file_name="verse_model.py")
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(agent)

    scenario.set_init_single("pump", init, (PumpMode.default,))
    traces = scenario.verify(duration, time_step)
    end_low = traces.root.trace["pump"][-2]
    end_high = traces.root.trace["pump"][-1]
    return [end_low[1:], end_high[1:]], traces


def simulate_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):

    agent = PumpAgent("pump", file_name="verse_model.py")
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))

    scenario.add_agent(agent)

    scenario.set_init_single("pump", init, (PumpMode.default,))
    traces = scenario.simulate(duration, time_step)
    end = traces.root.trace["pump"][-1]
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
        result, tree = verify_bolus(
            copy.deepcopy(prev_result), boluses[i][1][0], boluses[i][1][1], duration
        )
        link_nodes(prev_tree.root, tree.root)
        prev_result, prev_tree = result, tree
    return result1, tree1


def simulate_boluses(init, scenario):
    result1, tree1 = simulate_bolus(init, boluses[0][1], boluses[0][1], duration)
    prev_result, prev_tree = result1, tree1
    for i in range(1, len(boluses)):
        result, tree = simulate_bolus(
            copy.deepcopy(prev_result), boluses[i][1], boluses[i][1], duration
        )
        link_nodes(prev_tree.root, tree.root)
        prev_result, prev_tree = result, tree
    return result1, tree1


# TODO should this be an enum
def plot_variable(fig, tree, var, mode: Union["simulate", "verify"] = "simulate", show=True):
    idx = state_indices[var] + 1  # time is 0, so 1-index
    if mode == "verify":
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
    agent = PumpAgent(
        "pump", simulation_scenario=simulation_scenario, file_name=os.getcwd() + "/verse_model.py"
    )
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (PumpMode.default,))
    duration = simulation_scenario.simulation_duration
    time_step = 1
    traces = scenario.simulate(duration, time_step)
    return traces


def generate_all_three_meal_traces(
    init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs, trace_directory="traces/"
):

    all_combinations = np.array(
        np.meshgrid(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs)
    ).T.reshape(-1, 5)
    existing_files = os.listdir(trace_directory)
    for i in tqdm(range(len(all_combinations))):
        combination = all_combinations[i]
        tqdm.write(f"Simulating with init state {combination}")
        bg, br, bc, lc, dc = combination
        filename = f"trace_{bg}_{br}_{bc}_{lc}_{dc}.csv"
        if filename in existing_files:
            continue
        traces = simulate_three_meal_scenario(bg, br, bc, lc, dc)
        save_traces(traces, os.path.join(trace_directory, filename))


def plot_trace(filename, variable, trace_directory="traces/"):
    path = os.path.join(trace_directory, filename)
    df = pd.read_csv(path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df[variable]))
    fig.update_layout(dict(xaxis_title="t", yaxis_title=variable))
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
    scenario.set_init_single("pump", init, (PumpMode.default,))
    duration = simulation_scenario.simulation_duration
    time_step = 1
    traces = scenario.verify(duration, time_step)
    return traces


def save_traces(traces: AnalysisTree, filename):
    data = np.array(list(traces.root.trace.values())[0])  # we only have one agent
    cols = ["t"] + state_variable_names[:-1]  # drop the discrete state
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filename)
    print(data.shape)


if __name__ == "__main__":
    basal_rate = [0]

    traces = simulate_three_meal_scenario(130, 0, 60, 0, 0)
    save_traces(traces, "traces/trace_130_0_60_0_0.csv")
    plot_trace("trace_130_0_60_0_0.csv", "G")

# TODO plot traces only takes in a file right now-- make it support a trace object
# TODO you have to pass the traces dir into save_traces
