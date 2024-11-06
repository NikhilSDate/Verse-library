from enum import Enum, auto
import copy


class ThermoMode(Enum):
    A = auto()
    B = auto()


class State:

    # body model
    IGNORE: float # used to make the body model 1-indexed (should be fixed later)
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
    Isc1: float # subcutaneous insulin concentration; this needs to be touched directly
    Isc2: float # subcutaneous insulin concentration 2: don't touch directly
    Hsc1: float
    Hsc2: float
    G: float
    
    
    # scenario model
    # TODO: this should really not even be in the continuous state since these values are never going to change, but I don't know if there is a better way to model this
    D_1: float # carbs in first meal (in mg)
    t_1: float # time of first meal (minutes)
    D_2: float # carbs in second meal (in mg)
    t_2: float # time of second meal (minutes)
    D_3: float
    t_3: float

    # pump model
    pump_iob_0: float
    pump_elapsed_0: int
    pump_iob_1: float
    pump_elapsed_1: float
    pump_iob_2: float
    pump_elapsed_2: float
    pump_iob_3: float
    pump_elapsed_3: float
    pump_iob: float

    agent_mode: ThermoMode

    def __init__(self, x, agent_mode: ThermoMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output


if __name__ == "__main__":
    pass
