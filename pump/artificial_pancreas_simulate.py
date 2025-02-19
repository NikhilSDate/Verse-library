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
from hovorka_model import HovorkaModel, patient_original
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

##############
##############
##############


def simulate_multi_meal_scenario(init_bg, params, basal_iq, boluses, meals, duration=24 * 60, settings=None, logging=True, model_params='2004'):

    simulation_scenario = SimulationScenario(basal_iq, boluses, meals, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=basal_iq, settings=settings)
    body = HovorkaModel(params)
    cgm = CGM()
    if logging:
        logger = Logger('results/logs')
    else:
        logger = NotLogger()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
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

def verify_multi_meal_scenario(init_bg, params, basal_iq, boluses, meals, duration=24 * 60, settings=None, log_dir=None):
    meals_low, meals_high = meals # the actual meal objects will only be used for the meal times
    simulation_scenario = SimulationScenario(basal_iq, boluses, meals_low, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=basal_iq, settings=settings) # we don't have state stuff working yet, so disable basal IQ
    body = HovorkaModel(params)
    cgm = CGM()
    logger = Logger(log_dir=log_dir)
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
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
    
def get_recommended_settings(TDD=18.28, BW = 74.9):
    
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
    settings = get_recommended_settings(TDD=39.22)    
    BW = 74.9  # kg
    basal = 0  # units
    traces = simulate_multi_meal_scenario(120, BW, False, [Bolus(0, -1, BolusType.Simple, None)], [Meal(0, 100)], duration=8 * 60, settings=settings)
    fig1 = plot_variable(traces, 'GluMeas')
    breakpoint()
