from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
import plotly.graph_objects as go
import pickle

if __name__ == '__main__':
    with open('hidden/traces/thermostat_hidden_0.001.pkl', 'rb') as f:
        fig_hidden: go.Figure = pickle.load(f)
    with open('hidden/traces/thermostat_explicit_0.01.pkl', 'rb') as f:
        fig_explicit: go.Figure = pickle.load(f)    


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Assuming fig_hidden and fig_explicit are your existing figures
fig_hidden_traces = [trace for trace in fig_hidden.data]
fig_explicit_traces = [trace for trace in fig_explicit.data]

# Create a new figure with shared axes
combined_fig = make_subplots()

# Add all traces from fig_hidden with transparent red color
for trace in fig_hidden_traces:
    # For scatter plots
    if hasattr(trace, 'marker'):
        trace.marker.color = 'rgba(255, 0, 0, 0.1)'  # Transparent red
    # For line plots
    if hasattr(trace, 'line'):
        trace.line.color = 'rgba(255, 0, 0, 0.1)'  # Transparent red
    # For filled area plots
    if hasattr(trace, 'fillcolor'):
        trace.fillcolor = 'rgba(255, 0, 0, 0.1)'  # More transparent red for fill
    combined_fig.add_trace(trace)

# Add all traces from fig_explicit with transparent blue color
for trace in fig_explicit_traces:
    # For scatter plots
    if hasattr(trace, 'marker'):
        trace.marker.color = 'rgba(0, 0, 255, 0.1)'  # Transparent blue
    # For line plots
    if hasattr(trace, 'line'):
        trace.line.color = 'rgba(0, 0, 255, 0.1)'  # Transparent blue
    # For filled area plots
    if hasattr(trace, 'fillcolor'):
        trace.fillcolor = 'rgba(0, 0, 255, 0.1)'  # More transparent blue for fill
    combined_fig.add_trace(trace)

# Update layout
combined_fig.update_layout(
    title="Combined Figure with Shared Axes",
    xaxis_title=fig_hidden.layout.xaxis.title.text,
    yaxis_title=fig_hidden.layout.yaxis.title.text
)

# Ensure the legends differentiate between the two data sources
for i, trace in enumerate(combined_fig.data):
    if i < len(fig_hidden_traces):
        trace.name = f"Hidden: {trace.name}" if trace.name else f"Hidden Trace {i+1}"
    else:
        trace.name = f"Explicit: {trace.name}" if trace.name else f"Explicit Trace {i-len(fig_hidden_traces)+1}"

combined_fig.show()