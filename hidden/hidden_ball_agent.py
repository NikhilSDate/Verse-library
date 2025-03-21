# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from verse.plotter.plotter2D import *
import plotly.graph_objects as go


class BallAgent(BaseAgent):
    """Dynamics of a frictionless billiard ball
    on a 2D-plane"""

    def __init__(self, id, code=None, file_name=None):
        """Contructor for the agent
        EXACTLY one of the following should be given
        file_name: name of the decision logic (DL)
        code: pyhton string defning the decision logic (DL)
        """
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)

    def dynamics(self, t, state):
        """Defines the RHS of the ODE used to simulate trajectories"""
        x, y, vx, vy = state
        x_dot = vx
        y_dot = vy
        vx_dot = 0
        vy_dot = 0
        return [x_dot, y_dot, vx_dot, vy_dot]
    
    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map = None
    ):
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            
            x, y, vx, vy = init
            rx, ry, rvx, rvy = x, y, vx, vy

            cr = 0.85
            if x < 0:
                rvx = -vx*cr
                rx = 0
            if y < 0:
                rvy = -vy*cr
                ry = 0
            if x > 20:
                # TODO: Q. If I change this to ego.x >= 20 then the model does not work.
                # I suspect this is because the same transition can be take many, many times.
                # We need to figure out a clean solution
                rvx = -vx*cr
                rx = 20
            if y > 20:
                rvy = -vy*cr
                ry = 20     
            
            init = [rx, ry, rvx, rvy]
                   
            r = ode(self.dynamics)
            r.set_initial_value(init)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace
        pass


if __name__ == "__main__":
    aball = BallAgent(
        "red_ball", file_name="/Users/mitras/Dpp/GraphGeneration/demo/ball_bounces.py"
    )
    trace = aball.TC_simulate({"none"}, [5, 10, 2, 2], 10, 0.05)
    fig = simulation_tree(trace, map=None, fig=go.Figure(), x_dim = 1, y_dim = 2, print_dim_list=None, map_type='lines', scale_type='trace', label_mode='None', sample_rate=1)
    fig.show()
    print(trace)
