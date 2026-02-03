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
from artificial_pancreas_scenario import ResultType
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
    pump = InsulinPumpModel(simulation_scenario, settings=simulation_scenario.settings[0], trace=True) 
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
    try:
        traces = scenario.simulate(simulation_scenario.sim_duration, time_step)
        traces.metadata = agent.trace_metadata
        pump.write_trace(log_dir)
        return traces
    except Exception as e:
        print(e.with_traceback(None))
        pump.write_trace(log_dir)
        pass



def simulate_multi_meal_scenario(simulation_scenario: SimulationScenario, log_dir=None, params={}):
    trace = params.get('trace', False)
    pump = InsulinPumpModel(simulation_scenario, settings=simulation_scenario.settings, trace=trace) 
    body = HovorkaModel(simulation_scenario.params)
    cgm = CGM()
    logger = Logger(log_dir=log_dir)
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
    )
    try:
        init = agent.get_init_state(simulation_scenario.init_bg, simulation_scenario.get_meals(), simulation_scenario.settings, simulation_scenario.errors, get_cgm_error(simulation_scenario.cgm_config))
        scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
        scenario.add_agent(agent)
        scenario.set_init_single(
            "pump", [init, init], (PumpMode.default,)
        )  # TODO what's the other half of the tuple?

        time_step = simulation_scenario.time_step
        traces = scenario.simulate(simulation_scenario.sim_duration, time_step)    
        traces.metadata = agent.trace_metadata
        if trace and log_dir is not None:
            pump.write_trace(log_dir)
        return VerificationResult(ResultType.OK, traces)
    except Exception as e:
        print('here')
        raise e
        err_info = agent.get_error_info()
        err_info.e = e
        return VerificationResult(ResultType.ERROR, err_info)

def get_cgm_error_range(cgm_config: CGMConfig):
    error_low = [cgm_config.bias[0], cgm_config.offset[0]]
    error_high = [cgm_config.bias[1], cgm_config.offset[1]]
    return error_low, error_high

def get_cgm_error(cgm_config: CGMConfig):
    return [cgm_config.bias, cgm_config.offset]

# track_inits is a hack: if set to True, no actual verification will be performed, and the function will just return the initial values that DryVR chooses
def verify_multi_meal_scenario(simulation_scenario: SimulationScenario, params={}) -> VerificationResult:
    params = params if params is not None else {}
    log_dir = params.get('log_dir', None)
    sim_trace_num = params.get('sim_trace_num', 10)
    pump = InsulinPumpModel(simulation_scenario, settings=simulation_scenario.settings[0]) 
    body = HovorkaModel(simulation_scenario.params)
    cgm = CGM()
    logger = Logger(log_dir=log_dir)
    agent = ArtificialPancreasAgent(
        "pump", body, pump, cgm, simulation_scenario, logger, file_name=PUMP_PATH + "verse_model.py"
    )
    try:
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

        time_step = simulation_scenario.time_step
        traces = scenario.verify(simulation_scenario.sim_duration, time_step, params={'sim_trace_num': sim_trace_num})  
        traces.metadata = agent.trace_metadata
        return VerificationResult(ResultType.OK, traces)
    except Exception as e:
        err_info = agent.get_error_info()
        err_info.e = e
        return VerificationResult(ResultType.ERROR, err_info)


def evaluate_safety_constraint(traces, variable, safety_func):
    reachtube_trace = extract_variable(traces, variable)
    reachtube_safety = safety_func(reachtube_trace)
    
    sim_safety = np.array([True] * len(reachtube_safety))
    for sim in traces.root.sims:
        trace = extract_variable(sim, variable, type=TraceType.SIM)
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

def get_index(var):
    return state_indices[var] + 1

# TODO: fix this
def extract_variable(traces: AnalysisTree | np.ndarray, var: str, type=TraceType.VERIF, agent='pump'):
    if isinstance(traces, AnalysisTree):
        raw_trace = np.array(traces.root.trace[agent])
    else:
        raw_trace = traces
    index = get_index(var)
    if type == TraceType.SIM:
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
    # with open('./results/perfectly_unsafe/scenario_080000000071e0c19/scenario.pkl', 'rb') as f:
    #     scenario = pickle.load(f)
    # traces = verify_multi_meal_scenario(scenario)
    # fig = plot_variable(traces, 'G', show=False)
    # fig.write_image('extended_fixed.png')
    # print(evaluate_safety_constraint(traces, 'G', lambda glucose: AGP_safety(glucose))) # glucose shouldn't be >= 250 for > 30min
    settings = get_recommended_settings(TDD=39.22, BW=74.9)
    settings['basal_iq'] = False
    
    scenario1 = SimulationScenario(
        init_bg=120,
        boluses=[Bolus(0, 0, BolusType.Simple, 0, True, None)],
        meals=[Meal(0, 75, DEFAULT_MEAL)],
        errors=1,
        settings=settings,
        params=patient_original({'basalGlucose': 6.5}),
        cgm_config=CGMConfig(1, 0),
        sim_duration=1 * 60,
        time_step=1
    )

    scenario2 = SimulationScenario(
        init_bg=120,
        boluses=[Bolus(0, 0, BolusType.Simple, 0, True, None)],
        meals=[Meal(0, 75, DEFAULT_MEAL)],
        errors=1,
        settings=settings,
        params=patient_original({'basalGlucose': 6.5}),
        cgm_config=CGMConfig(1, 0),
        sim_duration=1 * 60,
        time_step=1
    )

    breakpoint()

    # (70, 180): True, True, False, False, False
    # (70, 100): True True, False, False, False
    
    # fig.write_image(f'figs/{t_max}_bolus.png') 

# {'tir': 0.8514920194309508, 'low': 92.32843681295014, 'high': 233.1487607240195}
# {'tir': 0.8507980569049272, 'low': 91.07826567567132, 'high': 234.21975753031126}
# {'tir': 0.8015267175572519, 'low': 59.83497960799967, 'high': 197.72905701924455}
# 