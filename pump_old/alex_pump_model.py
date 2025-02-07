from enum import Enum, auto
import copy


class ThermoMode(Enum):
    A = auto()
    B = auto()


class State:

    # body model
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
    Isc1: float
    Isc2: float
    Hsc1: float
    Hsc2: float
    Ra: float
    G: float

    ## glucose model
    #Gs: float  # mg/dL     subcutanous glucose concentration, RESULT
    #um: float  # mg/kg/min    glucose rate of appearance, RESULT
    #X: float  # pmol/L    insulin conc. in remote chamber/insulin in the interstitial fluid
    #Isc1: float  # pmol/kg   subcutaneous insulin in chamber 1 (liver)
    #Isc2: float  # pmol/kg   subcutaneous insulin in chamber 2 (periphery? plasma?)
    #Gt: float  # mg/dL     glucose conc. in rapidly equilibriating tissues/ glucose MASS in rapidly equilibriating tissues
    #Gp: float  # mg/dL     glucose conc. in plasma/ glucose MASS in plasma
    #Il: float  # pmol/kg   portal vein insulin mass/insulin mass in liver, I_po in Dalla Man
    #Ip: float  # pmol/kg   insulin mass in plasma
    #I1: float  # pmol/L    insulin chamber #1 concentration
    #Id: float  # pmol/L    delayed insulin from chamber 1/ delayed insulin signal realized with a chain of two compartments

    ## insulin model
    #um: float  # mg/kg/min    glucose rate of appearance, RESULT
    #Gs: float  # mg/dL     subcutanous glucose concentration, RESULT
    #qsto1: float  # mg        glucose in stomach (solid)
    #qsto2: float  # mg        glucose in stomach (liquid)
    #qgut: float  # mg        glucose in the intestines

    # pump model
    pump_iob_0: float
    pump_elapsed_0: int
    pump_iob_1: float
    pump_elapsed_1: float
    pump_iob_2: float
    pump_elapsed_2: float
    pump_iob: float

    agent_mode: ThermoMode

    def __init__(self, x, agent_mode: ThermoMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output


if __name__ == "__main__":
    pass
