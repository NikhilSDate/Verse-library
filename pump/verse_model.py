import copy

from enum import Enum, auto
from body_model import BodyModel


class PumpMode(Enum):
    default = auto()    

    
# State = body state + pump state + scenario state + derived state
class State:

    # Body model
    G: float               # Amount of glucose in compartment 1 [mmol]
    InsSub1: float               # Amount of glucose in compartment 2 [mmol]
    InsSub2: float               # Amount of insulin in compartment 1 [mU]
    InsPlas: float               # Amount of insulin in compartment 2 [mU]
    InsActT: float               # Amount of glucose in the main blood stream [mmol]
    InsActD: float               # Amount of glucose in peripheral tissues [mmol]
    InsActE : float               # Plasma insulin concentration [mU/L]
    GutAbs: float               # Insluin in muscle tissues [1], x1*Q1 = Insulin dependent uptake of glucose in muscles
    GluPlas: float               # [1], x2*Q2 = Insulin dependent disposal of glucose in the muscle cells
    GluComp: float               # Insulin in the liver [1], EGP_0*(1-x3) = Endogenous release of glucose by the liver
    GluInte: float     
    GluMeas: float
    
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
    time_out_of_range: float
    prediction_error: float
    
    # settings
    basal_rate: float    
    
    meal_0_error: float
    meal_1_error: float
    meal_2_error: float
    meal_3_error: float
    meal_4_error: float
    meal_5_error: float
    meal_6_error: float
    meal_7_error: float
    meal_8_error: float
    meal_9_error: float
    
    # experimental
    meal_0_time: float
    meal_1_time: float
    meal_2_time: float
    meal_3_time: float
    meal_4_time: float
    meal_5_time: float
    meal_6_time: float
    meal_7_time: float
    meal_8_time: float
    meal_9_time: float
    
    # CGM
    # cgm_error: float
    
    agent_mode: PumpMode

    def __init__(self, x, agent_mode: PumpMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    return output
