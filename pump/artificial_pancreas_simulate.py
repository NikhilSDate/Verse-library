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
import pickle
from safety.safety import tir_analysis

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


def simulate_multi_meal_scenario(init_bg, BW, basal_rate, boluses, meals, duration=24 * 60, settings=None):

    simulation_scenario = SimulationScenario(basal_rate, boluses, meals, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=True, settings=settings)
    body = HovorkaModel(BW, init_bg)
    cgm = CGM()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, file_name=PUMP_PATH + "verse_model.py"
    )
    init_state = agent.get_init_state(init_bg, meals)
    init = [init_state, init_state]  # TODO why twice?

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single(
        "pump", init, (PumpMode.default,)
    )  # TODO what's the other half of the tuple?

    time_step = 1
    traces = scenario.simulate(simulation_scenario.sim_duration, time_step)

    return traces

def verify_multi_meal_scenario(init_bg, BW, basal_rate, boluses, meals, duration=24 * 60, settings=None):
    meals_low, meals_high = meals # the actual meal objects will only be used for the meal times
    simulation_scenario = SimulationScenario(basal_rate, boluses, meals_low, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=False, settings=settings) # we don't have state stuff working yet, so disable basal IQ
    body = HovorkaModel(BW, init_bg)
    cgm = CGM()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, file_name=PUMP_PATH + "verse_model.py"
    )
    init_state = agent.get_init_range(init_bg[0], init_bg[1], meals_low, meals_high)
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

# t' = a * t + b
def linear_transform_trace(traces, agent, index, a, b):
    for i in range(len(traces.root.trace[agent])):
        traces.root.trace[agent][i][index] = a * traces.root.trace[agent][i][index] + b

def extract_variable(traces, agent, index):
    raw_trace = np.array(traces.root.trace[agent])
    return raw_trace.reshape((-1, 2, raw_trace.shape[1]))[:, :, index]

def plot_variable(tree, var, show=True, fig = None):
    if fig is None:
        fig = go.Figure()
    idx = state_indices[var] + 1  # time is 0, so 1-index
    if tree.root.type == AnalysisTreeNodeType.REACH_TUBE:
        fig = reachtube_tree(tree, None, fig, 0, idx)
    else:
        fig = simulation_tree(tree, None, fig, 0, idx)
    if show:
        fig.show()
    return fig

def iob_accuracy_test(settings, starting_bg=120, num_meals=10):
    BW = 70  # kg
    basal = 0  # units
    boluses = []
    boluses = []
    meals_low = []
    meals_high = []
    for i in range(num_meals): 
        boluses.append(Bolus(i * 60, 0, BolusType.Simple, None))
        meals_low.append(Meal(i * 60, 75))
        meals_high.append(Meal(i * 60, 100))
    traces = verify_multi_meal_scenario([120, 120], BW, basal, boluses, [meals_low, meals_high], duration=(num_meals + 5) * 60, settings=settings)
    linear_transform_trace(traces, 'pump', state_indices['iob'] + 1, 0.12 * 70, 0) # + 1 because time is index 0
    fig1 = plot_variable(traces, 'iob')
    fig2 = plot_variable(traces, 'I')
    fig3 = plot_variable(traces, 'iob_error')
    fig4 = plot_variable(traces, 'prediction_error')
    fig5 = plot_variable(traces, 'G')
    fig1.write_image('results/iob_verif.png')
    fig2.write_image('results/insulin_real_verif.png')
    fig3.write_image('results/iob_error_verif.png')
    fig4.write_image('results/prediction_error_verif.png')
    fig5.write_image('results/glucose_verif.png')
    breakpoint()
    
def get_recommended_settings(TDD = 39.2200, BW = 75):
    
     # according to the McGill simulator
    TDD = (0.75 * TDD + BW * 0.23) / 2
    TDB = TDD * 0.5
    rate = TDB / 24
    CF = 1700 / TDD
    carb_ratio = 450 / TDD
    duration = 5 # hours
    settings = {
        'carb_ratio': int(carb_ratio),
        'correction_factor': int(CF),
        'insulin_duration': int(duration * 60),
        'max_bolus': 10,
        'basal_rate': rate,
        'target_bg': 110
    }
    return settings


if __name__ == "__main__":
    settings = get_recommended_settings()
    print(settings)
    settings['insulin_duration'] = 150
    settings['basal_rate'] = 0.2
    BW = 75  # kg
    basal = 0  # units
    meals_low = [Meal(0, 50), Meal(240, 75)]
    meals_high = [Meal(0, 75), Meal(240, 100)]
    boluses = [Bolus(0, 0, BolusType.Simple, None), Bolus(240, 0, BolusType.Simple, None), Bolus(480, 0, BolusType.Simple, None)]
    traces = simulate_multi_meal_scenario(110, BW, basal, boluses, meals_low, duration=16 * 60, settings=settings)
    # glucose_trace = extract_variable(traces, 'pump', state_indices['G'] + 1)
    # print(tir_analysis(glucose_trace))
    fig = plot_variable(traces, 'G')
    # fig.write_image('results/bad_duration_glucose_.png')
    breakpoint()
