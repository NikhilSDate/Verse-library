from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
from pump.artificial_pancreas_scenario import Meal
from pump.simutils import DEFAULT_MEAL
from pump.hovorka_model import *

def parse_t1d_xml(xml_data):
    def parse_datetime(ts):
        return datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")

    def minutes_since_start(ts):
        return int((ts - t0).total_seconds() // 60)

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

    meals = []
    for (t, carbs) in M.items():
        meals.append(Meal(t, int(carbs), DEFAULT_MEAL))

    return G, M, I

if __name__ == '__main__':
    with open('/home/ndate/Research/OhioT1DM/2018/train/559-ws-training.xml') as f:
        data = f.read()

    G, M, I = parse_t1d_xml(data)
    param = patient_original({'basalGlucose': 6.5})
    model = HovorkaModel(param)
    model.set_meals(M)
    initial = model.get_init_state(G[0])
    f = model.model()

