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
    worst_case_count = 0
    best_case_count = 0
    for (low, high) in glucose_reachtube:
        if low >= tir_low and high <= tir_high:
            worst_case_count += 1
        if low <= tir_high and high >= tir_low:
            best_case_count += 1
    return best_case_count / len(glucose_reachtube), worst_case_count / len(glucose_reachtube)

if __name__ == '__main__':
    print(safety_report([100, 110, 100, 110, 10, 100]))