from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
from artificial_pancreas_scenario import Meal
from simutils import DEFAULT_MEAL
from hovorka_model import *
from scipy.integrate import ode
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List
from scipy.optimize import *
import pandas as pd
import pickle


@dataclass
class OhioT1DMTrace:
    G: Dict[int, float]
    M: List[Meal]
    I: defaultdict[int, float]
    duration: int

keys = ['basalGlucose', 'MCHO', 'w', 'TauS', 'EGP0', 'F01', 'k12', 'RTh', 'RCl', 'ka1', 'ka2', 'ka3', 'St', 'Sd', 'Se', 'ka', 'ke', 'Vi', 'Vg', 'Bio', 'TauM', 'TauGlu', 'TGlu', 'MCRGlu']

def parse_t1d_xml(xml_data, date=None, offset=0, duration=1440) -> OhioT1DMTrace:
    def parse_datetime(ts):
        return datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")

    def minutes_since_start(ts):
        return int((ts - t0).total_seconds() // 60)

    if date:
        date_start = datetime.strptime(date, "%d-%m-%Y")
    else:
        date_start = None

    root = ET.fromstring(xml_data)

    # Helper: Convert datetime to minutes since day start + offset
    def adjusted_minutes(ts):
        if date_start and ts < date_start:
            return None
        minutes = (ts - ts.replace(hour=0, minute=0, second=0)).total_seconds() // 60
        return int(minutes - offset)

    # Parse the XML
    root = ET.fromstring(xml_data)

    # Extract glucose events and establish time zero
    glucose_events = [(parse_datetime(e.attrib['ts']), float(e.attrib['value']))
                      for e in root.find('glucose_level')]
    if not glucose_events:
        raise ValueError("No glucose readings found")
    t0 = glucose_events[0][0]

    # Glucose: G[t]
    G = {}
    for ts, value in glucose_events:
        G[minutes_since_start(ts)] = value

    # Meal: M[t]
    M = defaultdict(float)
    for e in root.find('meal'):
        ts = parse_datetime(e.attrib['ts'])
        M[minutes_since_start(ts)] = float(e.attrib['carbs'])

    # Insulin: I[t] = bolus + basal + temp basal
    I = defaultdict(float)

    # Basal
    basal_events = [(parse_datetime(e.attrib['ts']), float(e.attrib['value']))
                    for e in root.find('basal')]
    basal_events.sort()
    for i in range(len(basal_events) - 1):
        start, rate = basal_events[i]
        end, _ = basal_events[i + 1]
        t_start = minutes_since_start(start)
        t_end = minutes_since_start(end)
        for t in range(t_start, t_end):
            I[t] += rate / 60

    # Extend final basal for up to 24 hours
    if basal_events:
        start, rate = basal_events[-1]
        t_start = minutes_since_start(start)
        for t in range(t_start, t_start + 24 * 60):
            I[t] += rate / 60

    # Temp basal overrides regular basal
    for e in root.find('temp_basal'):
        ts_begin = parse_datetime(e.attrib['ts_begin'])
        ts_end = parse_datetime(e.attrib['ts_end'])
        value = float(e.attrib['value'])
        for t in range(minutes_since_start(ts_begin), minutes_since_start(ts_end)):
            I[t] = value / 60

    # Bolus
    for e in root.find('bolus'):
        ts = parse_datetime(e.attrib['ts_begin'])
        I[minutes_since_start(ts)] += float(e.attrib['dose'])

    if date is not None:
        anchor_dt = datetime.strptime(date, "%d-%m-%Y") + timedelta(minutes=offset)
        anchor_t = int((anchor_dt - t0).total_seconds() // 60)

        # binary search would be more efficient but this is clearer
        anchor_t = min([t for t in G.keys() if t >= anchor_t])

        G = {t - anchor_t: v for t, v in G.items() if t >= anchor_t and t <= anchor_t + duration}
        M = {t - anchor_t: v for t, v in M.items() if t >= anchor_t and t <= anchor_t + duration}
        I = {t - anchor_t: v for t, v in I.items() if t >= anchor_t and t <= anchor_t + duration}
        I = defaultdict(float, I)


    meals = []
    for (t, carbs) in M.items():
        meals.append(Meal(t, int(carbs), 50))

    return OhioT1DMTrace(G, meals, I, duration)

def interpolate_glucose(G: Dict[int, float], duration: int):
    raw = np.full(shape=(duration,), fill_value=np.nan)
    for i in range(duration):
        if i in G:
            raw[i] = G[i]
    
    df = pd.Series(np.zeros(raw))
    df.interpolate(method='nearest')
    return df.to_numpy(dtype=float)

def split(variables):
    params = variables[:len(keys)]
    init = variables[len(keys):]
    return params, init

def run_model(variables, trace: OhioT1DMTrace, duration: int):

    print('here')

    G, M, I = trace.G, trace.M, trace.I
    
    M = [meal for meal in M if meal.time <= duration]

    params, init = split(variables)

    params = {keys[i]: params[i] for i in range(len(params))}
    params = patient_custom(params)

    # the rest of the variables are the initial value

    if params is None:
        # invalid params
        return None

    model = HovorkaModel(params)
    model.set_meals(M)
    state_vec = init

    time_step = 1

    predicted = np.zeros((duration,))
    
    for t in range(0, duration, time_step):  
        predicted[t] = model.mmol_to_mgdl(state_vec[-2])
        dose = I[t]
        r = ode(lambda s, state: model.model(t + s, state, dose))
        r.set_initial_value(state_vec)
        res: np.ndarray = r.integrate(r.t + time_step)
        state_vec = res.flatten()
    return predicted

# returns the mean squared error for a period of duration minutes
def harness(variables, trace: OhioT1DMTrace, duration: int) -> float:
    G = trace.G
    predicted = run_model(variables, trace, duration)
    if predicted is None:
        return np.inf
    error = 0
    count = 0
    for t in range(0, duration):
        if t not in G:
            continue
        error += (predicted[t] - G[t]) ** 2
        count += 1
    return error / count

def plot_predictions(variables, trace: OhioT1DMTrace, duration: int, path: str):
    
    G, M, I = trace.G, trace.M, trace.I
    predicted = run_model(variables, trace, duration)

    real_t = []
    real_G = []
    for t in range(0, duration):
        if t in G:
            real_t.append(t)
            real_G.append(G[t])

    plt.plot(real_t, real_G)
    plt.plot(np.arange(duration), [predicted[t] for t in range(duration)])

    plt.savefig(path)

# TODO: look at 575 more

def fit_params(trace, duration):
    initial = patient_original({'basalGlucose': 6.5})
    initial['basalGlucose'] = 6.5

    model = HovorkaModel(initial)
    state = model.get_init_state(trace.G[0])

    initial_vec = np.zeros(len(keys) + len(state))

    idx = 0
    for i in range(len(keys)):
        initial_vec[idx] = initial[keys[i]]
        idx += 1

    for i in range(len(state)):
        initial_vec[idx] = state[i]
        idx += 1
    
    plot_predictions(initial_vec, trace, duration, 'before.png')

    print(f'initial error: {harness(initial_vec, trace, duration)}')

    bounds = [(var * 0.2, var * 4) for var in initial_vec]

    objective_func = lambda x: harness(x, trace, duration)
    result = minimize(objective_func, initial_vec, bounds=bounds, tol=1e-3)
    
    optimal = result.x


    plot_predictions(optimal, trace, duration, 'after.png')

    return result.x[:len(keys)]

def evaluate_fit(params: np.ndarray, trace: OhioT1DMTrace, duration: int):
    def state_only_objective(state):
        variables = np.hstack([params, state])
        return harness(variables, trace, duration)

    def state_only_plot(state, path):
        variables = np.hstack([params, state])
        plot_predictions(variables, trace, duration, path)
    
    initial = patient_original({'basalGlucose': 6.5})
    initial['basalGlucose'] = 6.5

    model = HovorkaModel(initial)
    state = model.get_init_state(trace.G[0])

    duration = 1440

    initial_vec = np.array(state)

    state_only_plot(initial_vec, 'before.png')

    print(f'initial error: {state_only_objective(initial_vec)}')

    bounds = [(var * 0.2, var * 4) for var in initial_vec]

    objective_func = state_only_objective
    result = minimize(objective_func, initial_vec, bounds=bounds, tol=1e-3)
    
    optimal = result.x

    state_only_plot(optimal, 'after.png')
    breakpoint()


if __name__ == '__main__':
    with open('/home/ndate/Research/OhioT1DM/2020/train/552-ws-training.xml') as f:
        data = f.read()
    trace = parse_t1d_xml(data, date='17-04-2025', offset=360)

    test = parse_t1d_xml(data, date='18-04-2025', offset=360)

    # print(test.G[0])

    # params = fit_params(trace, 1440)

    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)
    
    evaluate_fit(params, test, 1440)