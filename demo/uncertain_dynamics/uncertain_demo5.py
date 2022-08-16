
from uncertain_agents import Agent5
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.plotter.plotter2D_old import plot_reachtube_tree

import matplotlib.pyplot as plt  
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Default = auto()


if __name__ == "__main__":
    scenario = Scenario()

    car = Agent5('car1')
    scenario.add_agent(car)
    # car = vanderpol_agent('car2', file_name=input_code_name)
    # scenario.add_agent(car)
    # scenario.set_sensor(FakeSensor2())
    # modify mode list input
    scenario.set_init(
        [
             [[1,1,1,0], [1.5,1.5,1,0]],
        ],
        [
            tuple([AgentMode.Default]),
        ],
        uncertain_param_list=[
            [[-0.5],[0.5]],
        ]
    )
    traces = scenario.verify(10, 0.01, reachability_method='MIXMONO_CONT')
    fig = plt.figure(0)
    fig = plot_reachtube_tree(traces.root, 'car1', 0, [1],fig=fig)
    fig = plt.figure(1)
    fig = plot_reachtube_tree(traces.root, 'car1', 0, [2],fig=fig)
    fig = plt.figure(2)
    fig = plot_reachtube_tree(traces.root, 'car1', 0, [3],fig=fig)
    fig = plt.figure(3)
    fig = plot_reachtube_tree(traces.root, 'car1', 0, [4],fig=fig)
    plt.show()
    # fig = go.Figure()
    # fig = simulation_tree(traces, None, fig, 1, 2,
    #                       'lines', 'trace', print_dim_list=[1, 2])
    # fig.show()