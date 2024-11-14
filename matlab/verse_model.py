import copy

from enum import Enum, auto
from body_model import BodyModel

class PumpMode(Enum):
    default = auto()

# TODO split into 3 subvectors -- body, scenario, pump
class State:

    # Body model
    G: float  # used to make the body model 1-indexed (should be fixed later)
    Gp: float
    Gt: float
    Il: float
    Ip: float
    I1: float
    Id: float
    Qsto1: float
    Qsto2: float
    Qgut: float
    X: float
    SRsH: float
    H: float
    XH: float
    Isc1: float  # Insert insulin here when acting as insulin pump
    Isc2: float
    Hsc1: float
    Hsc2: float

    # Scenario model
    # TODO: this should really not even be in the continuous state since these values are never going to change, but I don't know if there is a better way to model this
    D_1: float  # carbs in first meal (in mg)
    t_1: float  # time of first meal (minutes)
    D_2: float  # carbs in second meal (in mg)
    t_2: float  # time of second meal (minutes)
    D_3: float
    t_3: float

    # Pump model
    pump_iob_0: float
    pump_elapsed_0: int
    pump_iob_1: float
    pump_elapsed_1: float
    pump_iob_2: float
    pump_elapsed_2: float
    pump_iob_3: float
    pump_elapsed_3: float
    pump_iob: float

    agent_mode: PumpMode

    def __init__(self, x, agent_mode: PumpMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output
