from origin_agent import thermo_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario.scenario import ScenarioConfig
from verse.sensor.example_sensor.thermo_sensor import ThermoSensor
import plotly.graph_objects as go
from enum import Enum, auto
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *


class ThermoMode(Enum):
    ON = auto()
    OFF = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/thermo_controller.py"
    config = ScenarioConfig(parallel=False)
    scenario = Scenario(config)

    car = thermo_agent("test", file_name=input_code_name)
    scenario.add_agent(car)
    # car = thermo_agent("test2", file_name=input_code_name)
    # scenario.add_agent(car)
    # modify mode list input
    scenario.set_init(
        [
            [[75.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
            # [[76.0, 0.0, 0.0], [76.0, 0.0, 0.0]],
        ],
        [
            tuple([ThermoMode.ON]),
            # tuple([ThermoMode.ON]),
        ],
    )
    traces = scenario.verify(20, 0.1)
    fig = go.Figure()
    fig = reachtube_tree(traces, None, fig, 2, 1, [2, 1], "lines", "trace")
    fig.show()
