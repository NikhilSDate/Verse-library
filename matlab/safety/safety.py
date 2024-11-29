import numpy as np

# assume bg trace is sampled at regular intervals
def safety_report(traces, range_low=70, range_high=180, critical_low=40, critical_high=250, agent_name='pump'):
    bg_trace = np.array(traces.root.trace[agent_name])[:, 1]
    N = len(bg_trace)
    TIR = np.count_nonzero((bg_trace >= range_low) & (bg_trace <= range_high)) / N
    time_critical_low = np.count_nonzero(bg_trace <= critical_low) / N
    time_critical_high = np.count_nonzero(bg_trace >= critical_high) / N
    return TIR, time_critical_low, time_critical_high

if __name__ == '__main__':
    print(safety_report([100, 110, 100, 110, 10, 100]))