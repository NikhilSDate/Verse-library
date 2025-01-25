import copy

from enum import Enum, auto
from body_model import BodyModel


class PumpMode(Enum):
    default = auto()

# State = body state + pump state + scenario state + derived state
class State:

    # Body model
    D1: float               # Amount of glucose in compartment 1 [mmol]
    D2: float               # Amount of glucose in compartment 2 [mmol]
    S1: float               # Amount of insulin in compartment 1 [mU]
    S2: float               # Amount of insulin in compartment 2 [mU]
    Q1: float               # Amount of glucose in the main blood stream [mmol]
    Q2: float               # Amount of glucose in peripheral tissues [mmol]
    I : float               # Plasma insulin concentration [mU/L]
    x1: float               # Insluin in muscle tissues [1], x1*Q1 = Insulin dependent uptake of glucose in muscles
    x2: float               # [1], x2*Q2 = Insulin dependent disposal of glucose in the muscle cells
    x3: float               # Insulin in the liver [1], EGP_0*(1-x3) = Endogenous release of glucose by the liver
    C: float     
    G: float
    
    # pump state
    iob: float
    
    # scenario state
    carbs_0: float
    carbs_1: float
    carbs_2: float
    carbs_3: float
    carbs_4: float
    carbs_5: float
    carbs_6: float
    carbs_7: float
    carbs_8: float
    carbs_9: float
    
    # derived state
    iob_error: float
    
    # TODO verification will vary the meal amounts so it might actually need to go back in here
    # TODO if you want to chain multiple verifications together or if you want to verify the pump state (IOB)

    agent_mode: PumpMode

    def __init__(self, x, agent_mode: PumpMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output
