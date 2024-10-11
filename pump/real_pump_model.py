from enum import Enum, auto
import copy

class ThermoMode(Enum):
    A=auto()
    B=auto()

class State:
    # body model
    Gs: float # subcutanous glucose concentration ??? (is this plasma glucose concentration mg/dL)
    X: float # insulin conc. in remote chamber/insulin in the interstitial fluid: pmol/L
    Isc1: float # subcutaneous insulin in chamber 1: pmol/kg?
    Isc2: float # subcutaneous insulin in chamber 2 pmol/kg?
    Gt: float # Glcuose conc. in rapidly equilibriating tissues/ glucose MASS in rapidly equilibriating tissues: mg/kg
    Gp: float # Glucose conc. in plasma/ glucose MASS in plasma: mg/kg
    Il: float # Portal vein insulin mass/insulin mass in liver.: pmol/kg
    Ip: float # Insulin mass in plasma: pmol/kg
    I1: float # Insulin chamber #1 concentration: pmol/L
    Id: float # delayed insulin from chamber 1/ delayed insulin signal realized with a chain of two compartments: pmol/L
    
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