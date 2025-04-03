import numpy as np

# assume bg trace is sampled at regular intervals
def safety_report(traces, range_low=70, range_high=180, critical_low=40, critical_high=250, agent_name='pump', G_idx=11):
    bg_trace = np.array(traces.root.trace[agent_name])[:, G_idx + 1]
    N = len(bg_trace)
    TIR = np.count_nonzero((bg_trace >= range_low) & (bg_trace <= range_high)) / N
    time_critical_low = np.count_nonzero(bg_trace <= critical_low) / N
    time_critical_high = np.count_nonzero(bg_trace >= critical_high) / N
    return TIR, time_critical_low, time_critical_high

# glucose reachtube is array of [low, high] indexed by time
def tir_analysis(glucose_reachtube, tir_low=70, tir_high=180):
    hlb = np.max(glucose_reachtube[:, 1])
    lub = np.min(glucose_reachtube[:, 1])
    worst_case_count = 0
    best_case_count = 0
    for (low, high) in glucose_reachtube:
        
        if low >= tir_low and high <= tir_high:
            worst_case_count += 1
        if low <= tir_high and high >= tir_low:
            best_case_count += 1
        
    return {
        'tir': 
            {
                'low': worst_case_count / len(glucose_reachtube),
                'high': best_case_count / len(glucose_reachtube)
            },
        'lub': lub,
        'hlb': hlb
    }


def range_bounds(glucose_reachtube, lb, ub, relative=True):
    '''
    returns bounds on fraction of reachtube that's in the half-open interval [lb, ub)
    '''
    low_count = 0
    high_count = 0
    for (low, high) in glucose_reachtube:
        if low >= lb and high < ub:
            low_count += 1
        if low < ub and high >= lb:
            high_count += 1
    if relative:
        return (low_count / len(glucose_reachtube), high_count / len(glucose_reachtube))
    else:
        return (low_count, high_count)
    
def AGP_report(glucose_reachtube, AGP_config=[(-np.inf, 54), (54, 70), (70, 180), (180, 250), (250, np.inf)]):
    percs = []
    for (low, high) in AGP_config:
        percs.append(range_bounds(glucose_reachtube, low, high))
    return percs

def AGP_safety(glucose_reachtube, targets=[0.01, 0.04, 0.70, 0.25, 0.05]):
    '''
    targets 0, 1, 3, and 4 are upper bounds
    target 2 (for the ideal range) is a lower bound
    '''
    report = AGP_report(glucose_reachtube)
    lower_bounds_ranges = [0, 1, 3, 4]
    results = [True] * len(targets)
    for idx in lower_bounds_ranges:
        if report[idx][1] >= targets[idx]:
            results[idx] = False
    upper_bound_ranges = [2]
    for idx in upper_bound_ranges:
        if report[idx][0] < targets[idx]:
            results[idx] = False
    return results

def range_time_safety(glucose_reachtube, range, time_bound):
    '''
    constraint of the form: glucose is within range for at most time_bound minutes
    '''
    low, high = range_bounds(glucose_reachtube, range[0], range[1], )
    return high <= time_bound
    
    
def tir_analysis_simulate(glucose_trace, tir_low=70, tir_high=180):
    low = np.min(glucose_trace)
    high = np.max(glucose_trace)
    count = 0
    for g in glucose_trace:
        if tir_low <= g <= tir_high:
            count += 1
    return {
        'tir': count / len(glucose_trace),
        'low': low,
        'high': high
    }
    
def realism(scenario, f_low=3, f_high=6):
    # two things controlling realism score: 
    
    # following Table 1 here: https://pmc.ncbi.nlm.nih.gov/articles/PMC6566372/
    
    meals_low = scenario["meals"][0]
    meals_high = scenario["meals"][1]
    
    weight = scenario["patient"]["w"]
        
    thresh_low = f_low * weight
    thresh_high = f_high * weight
    
    carbs_low = sum([meal.carbs for meal in meals_low])
    carbs_high = sum([meal.carbs for meal in meals_high])

    low_distance = max(0, thresh_low - carbs_low)
    high_distance = max(0, carbs_high - thresh_high)

    return {'carbs_high': carbs_high, 'carbs_low': carbs_low, 'low_distance': low_distance, 'high_distance': high_distance}
    


if __name__ == '__main__':
    print(safety_report([100, 110, 100, 110, 10, 100]))