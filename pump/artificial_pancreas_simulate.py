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
from safety.safety import tir_analysis, tir_analysis_simulate, AGP_safety, range_time_safety
from simutils import *

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


def simulate_from_init(simulation_scenario: SimulationScenario, init, logging=False, log_dir=''):
    pump = InsulinPumpModel(simulation_scenario, settings=simulation_scenario.settings[0]) 
    body = HovorkaModel(simulation_scenario.params)
    cgm = CGM()
    if logging:
        logger = Logger(log_dir=log_dir)
    else:
        logger = NotLogger()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
    )
    init = [init, init]
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single(
        "pump", init, (PumpMode.default,)
    )  # TODO what's the other half of the tuple?

    time_step = 1
    traces = scenario.simulate(simulation_scenario.sim_duration, time_step)
    return traces


def simulate_multi_meal_scenario(init_bg, params, basal_iq, boluses, meals, errors=None, duration=24 * 60, settings=None, logging=True, model_params='2004', cgm_error=False):

    simulation_scenario = SimulationScenario(basal_iq, boluses, meals, sim_duration=duration)
    pump = InsulinPumpModel(simulation_scenario, basal_iq=basal_iq, settings=settings)
    body = HovorkaModel(params)
    if not cgm_error:
        cgm = CGM()
    else:
        cgm = VettorettiCGM({'start_day': 5})
    if logging:
        logger = Logger('results/logs')
    else:
        logger = NotLogger()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
    )
    init_state = agent.get_init_state(init_bg, meals, settings, errors)
    init = [init_state, init_state]  # TODO why twice?

    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single(
        "pump", init, (PumpMode.default,)
    )  # TODO what's the other half of the tuple?

    time_step = 1
    traces = scenario.simulate(simulation_scenario.sim_duration, time_step)

    return traces

def get_cgm_error_range(cgm_config: CGMConfig):
    error_low = [cgm_config.bias[0], cgm_config.offset[0]]
    error_high = [cgm_config.bias[1], cgm_config.offset[1]]
    return error_low, error_high


# TODO: change this so that it takes a SimulationScenario object directly, instead of the current arguments
# That's a much cleaner abstraction
# track_inits is a hack: if set to True, no actual verification will be performed, and the function will just return the initial values that DryVR chooses
def verify_multi_meal_scenario(simulation_scenario: SimulationScenario, log_dir='./debug', logging=False):
    pump = InsulinPumpModel(simulation_scenario, settings=simulation_scenario.settings[0]) 
    body = HovorkaModel(simulation_scenario.params)
    cgm = CGM()
    if logging:
        logger = Logger(log_dir=log_dir)
    else:
        logger = NotLogger()
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
    )
    settings_low, settings_high = simulation_scenario.settings
    errors_low, errors_high = simulation_scenario.errors
    meals_low, meals_high = get_meal_range(simulation_scenario.get_meals())
    cgm_low, cgm_high = get_cgm_error_range(simulation_scenario.cgm_config)
    init = agent.get_init_range(simulation_scenario.init_bg[0], simulation_scenario.init_bg[1], meals_low, meals_high, settings_low, settings_high, errors_low, errors_high, cgm_low, cgm_high)    
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single(
        "pump", init, (PumpMode.default,)
    )  # TODO what's the other half of the tuple?

    time_step = 1
    traces = scenario.verify(simulation_scenario.sim_duration, time_step)
    return traces

def evaluate_safety_constraint(traces, variable, safety_func):
    reachtube_trace = extract_variable(traces, 'pump', state_indices[variable] + 1)
    reachtube_safety = safety_func(reachtube_trace)
    
    sim_safety = np.array([True] * len(reachtube_safety))
    for sim in traces.root.sims:
        trace = extract_variable(sim, 'pump', state_indices[variable] + 1,simulate=True, raw=True)
        trace = np.column_stack((trace, trace))
        sim_safety = np.logical_and(sim_safety, safety_func(trace))
    
    result = [None] * len(reachtube_safety)
    for i in range(len(reachtube_safety)):
        if reachtube_safety[i]:
            result[i] = True
            assert(sim_safety[i])
        if not sim_safety[i]:
            result[i] = False
    return result
    


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

def extract_variable(traces, agent, index, simulate=False, raw=False):
    if not raw:
        raw_trace = np.array(traces.root.trace[agent])
    else:
        raw_trace = traces
    if simulate:
        return raw_trace.reshape((-1, raw_trace.shape[1]))[:, index]
    else:
        return raw_trace.reshape((-1, 2, raw_trace.shape[1]))[:, :, index]

def plot_variable(tree, var, show=True, fig = None, color='red'):
    if fig is None:
        fig = go.Figure()
    idx = state_indices[var] + 1  # time is 0, so 1-index
    if tree.root.type == AnalysisTreeNodeType.REACH_TUBE:
        fig = reachtube_tree(tree, None, fig, 0, idx)
    else:
        fig = simulation_tree(tree, None, fig, 0, idx, plot_color=[[color]])
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
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
    
def get_recommended_settings(TDD, BW, MDI=False):
    
    # TDD is already the pump TDD?
    if MDI:
        BW = BW * 2.20462
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
    with open('./results/perfectly_unsafe/scenario_080000000071e0c19/scenario.pkl', 'rb') as f:
        scenario = pickle.load(f)
    traces = verify_multi_meal_scenario(scenario)
    fig = plot_variable(traces, 'G', show=False)
    fig.write_image('extended_fixed.png')
    print(evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))) # glucose shouldn't be >= 250 for > 30min
    
    
    # (70, 180): True, True, False, False, False
    # (70, 100): True True, False, False, False
    
    # fig.write_image(f'figs/{t_max}_bolus.png') 

# {'tir': 0.8514920194309508, 'low': 92.32843681295014, 'high': 233.1487607240195}
# {'tir': 0.8507980569049272, 'low': 91.07826567567132, 'high': 234.21975753031126}
# {'tir': 0.8015267175572519, 'low': 59.83497960799967, 'high': 197.72905701924455}
# 