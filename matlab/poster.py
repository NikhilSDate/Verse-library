from real_pump_matlab import simulate_three_meal_scenario, plot_variable, verify_three_meal_scenario
import plotly.graph_objects as go

def generate_simulate_image():
    traces = simulate_three_meal_scenario(110, 0, 50, 100, 100)
    fig = plot_variable(go.Figure(), traces, 'G', 'simulate', show=False)
    fig.update_layout(
    xaxis=dict(
        title=dict(
            text="Time (minutes)",
            font=dict(size=12)
        )
    ),
    yaxis=dict(
        title=dict(
            text="Plasma Glucose (mg/dL)",
            font=dict(size=12)
        )
    ))
    fig.update_layout(
        xaxis_tickfont_size=12,  # Set font size for x-axis ticks
        yaxis_tickfont_size=12   # Set font size for y-axis ticks
    )
    return traces,fig
    
def generate_verify_figure():
    traces = simulate_three_meal_scenario(110, 0, 50, 100, 100)
    fig = plot_variable(go.Figure(), traces, 'G', 'simulate', show=False)
    fig.update_layout(
    xaxis=dict(
        title=dict(
            text="Time (minutes)",
            font=dict(size=12)
        )
    ),
    yaxis=dict(
        title=dict(
            text="Plasma Glucose (mg/dL)",
            font=dict(size=12)
        )
    ))
    fig.update_layout(
        xaxis_tickfont_size=12,  # Set font size for x-axis ticks
        yaxis_tickfont_size=12   # Set font size for y-axis ticks
    )
    return fig

if __name__ == "__main__":
    _, fig = generate_simulate_image()
    fig.show()
    breakpoint()