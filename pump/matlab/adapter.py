from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import matlab.engine

@dataclass(eq=True)
class MatlabData:
    meals: List[dict]
    infusions: Dict[int, float]
    options: Dict
    scenario: Tuple[int, int] # (duration, time_step)]
    trace: np.ndarray


def run_matlab_simulation(engine: matlab.engine.MatlabEngine, data: MatlabData) -> Tuple[MatlabData, str]:
    """
    Run the MATLAB simulation with the provided data.

    :param engine: The MATLAB engine instance.
    :param data: The data to be passed to the MATLAB simulation.
    :return: A tuple containing the result data and a status message.
    """
    try:
        # Convert Python data structures to MATLAB compatible types
        meals = matlab.double([list(meal.values()) for meal in data.meals])
        infusions = matlab.double(list(data.infusions.values()))
        options = matlab.double(list(data.options.values()))
        scenario = matlab.double(data.scenario)
        trace = matlab.double(data.trace.tolist())

        # Call the MATLAB function
        result = engine.runSimulation(meals, infusions, options, scenario, trace, nargout=2)

        # Convert result back to Python data structure
        result_data = MatlabData(
            meals=[dict(zip(['time', 'carbs'], meal)) for meal in result[0]],
            infusions=dict(zip(range(len(result[1])), result[1])),
            options=data.options,
            scenario=data.scenario,
            trace=np.array(result[2])
        )
        
        return result_data, "Simulation completed successfully."
    
    except Exception as e:
        return None, f"Error during simulation: {str(e)}"