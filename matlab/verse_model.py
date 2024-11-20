import copy

from enum import Enum, auto
from body_model import BodyModel


class PumpMode(Enum):
    default = auto()


class State:

    # Body model
    G: float
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

    # TODO verification will vary the meal amounts so it might actually need to go back in here
    # TODO if you want to chain multiple verifications together or if you want to verify the pump state (IOB)

    agent_mode: PumpMode

    def __init__(self, x, agent_mode: PumpMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output
