import sys

from typing import *

import pandas as pd
from dotenv import load_dotenv

import plotly.graph_objects as go
from verse import BaseAgent, Scenario, ScenarioConfig
from verse.analysis import AnalysisTreeNode, AnalysisTree, AnalysisTreeNodeType
from verse.plotter.plotter2D import reachtube_tree, simulation_tree

from verse_model import *
from artificial_pancreas_agent import *
from pump_model import *
from cgm import *
from hovorka_model import HovorkaModel

load_dotenv()
PUMP_PATH = os.environ["PUMP_PATH"]
TRACES_PATH = os.environ["TRACES_PATH"]

"""
SCENARIO: PUMP'S TARGET BG NOT EQUAL TO BODY'S BASAL BG
"""

# TODO translate verify() methods
# TODO make sure multiple meal/bolus work


def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):

    agent = ArtificialPancreasAgent("pump", file_name="verse_model.py")
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (PumpMode.default,))
    traces = scenario.verify(duration, time_step)
    end_low = traces.root.trace["pump"][-2]
    end_high = traces.root.trace["pump"][-1]
    return [end_low[1:], end_high[1:]], traces


def simulate_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):

    agent = ArtificialPancreasAgent("pump", file_name="verse_model.py")
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

def generate_all_three_meal_traces(
    init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs, trace_directory=TRACES_PATH
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


def verify_three_meal_scenario(init_bg, BW, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs):
    meals_low = [(breakfast_carbs[0], 0), (lunch_carbs[0], 240), (dinner_carbs[0], 660)]
    meals_high = [(breakfast_carbs[1], 0), (lunch_carbs[1], 240), (dinner_carbs[1], 660)]
    init_state_low = get_init_state(init_bg[0], BW, meals_low)
    init_state_high = get_init_state(init_bg[1], BW, meals_high)
    init = [init_state_low, init_state_high]
    simulation_scenario = SimulationScenario(meals_low, basal_rate)
    agent = ArtificialPancreasAgent(
        "pump", simulation_scenario=simulation_scenario, file_name=PUMP_PATH + "verse_model.py"
    )
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (PumpMode.default,))
    duration = simulation_scenario.simulation_duration
    time_step = 1
    traces = scenario.verify(duration, time_step)
    return traces


##############
##############
##############


def simulate_multi_meal_scenario(init_bg, BW, basal_rate, boluses, meals, duration=24 * 60):

    simulation_scenario = SimulationScenario(basal_rate, boluses, meals, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=True)
    body = HovorkaModel(BW, init_bg)
    cgm = CGM()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, file_name=PUMP_PATH + "verse_model.py"
    )
    init_state = agent.get_init_state(init_bg)
    init = [init_state, init_state]  # TODO why twice?

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single(
        "pump", init, (PumpMode.default,)
    )  # TODO what's the other half of the tuple?

    time_step = 1
    traces = scenario.simulate(simulation_scenario.sim_duration, time_step)

    return traces

def verify_multi_meal_scenario(init_bg, BW, basal_rate, boluses, meals, duration=24 * 60):

    simulation_scenario = SimulationScenario(basal_rate, boluses, meals, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=False) # we don't have state stuff working yet, so disable basal IQ
    body = HovorkaModel(BW, init_bg)
    cgm = CGM()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, file_name=PUMP_PATH + "verse_model.py"
    )
    init_state = agent.get_init_range(init_bg[0], init_bg[1])
    print(init_state)
    init = init_state

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single(
        "pump", init, (PumpMode.default,)
    )  # TODO what's the other half of the tuple?

    time_step = 1
    traces = scenario.verify(simulation_scenario.sim_duration, time_step)
    return traces


def save_traces(traces: AnalysisTree, filename, trace_directory=TRACES_PATH):
    data = np.array(list(traces.root.trace.values())[0])  # we only have one agent

    # TODO better way to get var_names
    var_names = [
        "D1",
        "D2",
        "S1",
        "S2",
        "Q1",
        "Q2",
        "I" ,
        "x1",
        "x2",
        "x3",
        "C",
        "G"
    ]
    cols = ["t"] + var_names
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(trace_directory + filename)


def plot_trace(filename, variable, trace_directory=TRACES_PATH):
    path = os.path.join(trace_directory, filename)
    df = pd.read_csv(path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df[variable]))
    fig.update_layout(dict(xaxis_title="t", yaxis_title=variable))
    fig.show()
    
def plot_variable(tree, var, mode: Union["simulate", "verify"] = "simulate", show=True):
    fig = go.Figure()
    idx = state_indices[var] + 1  # time is 0, so 1-index
    if mode == "verify":
        fig = reachtube_tree(tree, None, fig, 0, idx)
    else:
        fig = simulation_tree(tree, None, fig, 0, idx)
    if show:
        fig.show()
    return fig


if __name__ == "__main__":

    # TODO allow these to be passed in
    BW = 70  # kg
    basal = 0  # units
    boluses = [Bolus(0, 60, BolusType.Simple, None)]
    meals = [Meal(0, 60)]
    traces = verify_multi_meal_scenario([105, 120], BW, basal, boluses, meals, duration=24 * 60)
    breakpoint()
    plot_variable(traces, 'Q1', 'verify')
    breakpoint()

