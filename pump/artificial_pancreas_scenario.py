from typing import List, Union, Tuple, Dict
from enum import Enum
import os
from dataclasses import dataclass
import dataclasses
import numpy as np
from state_utils import *

class BolusType(str, Enum):
    Simple = 'Simple'
    Extended = 'Extended'

# TODO should we ever model time as a float?

@dataclass(eq=True, frozen=True)
class ExtendedBolusConfig:
    deliver_now_perc: float
    duration: float

@dataclass(eq=True, frozen=True)
class Bolus:
    time: int
    carbs: Union[int, Tuple[int, int]]
    type: BolusType
    meal_index: int 
    correction: bool
    config: ExtendedBolusConfig
    relative: bool

@dataclass(eq=True, frozen=True)
class Meal:
    time: int
    carbs: Union[int, Tuple[int, int]]
    TauM: float
    
@dataclass(eq=True, frozen=True)
class CGMConfig:
    bias: Union[float, Tuple[float, float]]
    offset: Union[float, Tuple[float, float]]
    
@dataclass(eq=True)
class ScenarioData:
    init_bg: Tuple[int, int]
    meals: List[Meal]
    boluses: List[Bolus]
    errors: List[float]
    settings: Dict
    params: Dict
    cgm_config: CGMConfig
    sim_duration: int
    

    
def get_meal_range(meals: List[Meal]):
    meals_low = []
    meals_high = []
    
    for m in meals:
        meals_low.append(dataclasses.replace(m, carbs=m.carbs[0], time=m.time[0]))
        meals_high.append(dataclasses.replace(m, carbs=m.carbs[1], time=m.time[1]))
    return (meals_low, meals_high)

def get_bolus_config(bolus: Bolus):
    return (bolus.type, bolus.config)

def set_bolus_config(bolus: Bolus, config: Tuple[BolusType, ExtendedBolusConfig]):
    return dataclasses.replace(bolus, type=config[0], config=config[1])

class SimulationScenario:

    # ==[ Instance Variables ]==
    # boluses (List[Bolus]) - list of Bolus objects to be administered
    #    meals (List[Meal]) - list of Meal objects to be administered
    #           iob (float) - insulin on board in units
    #    basal_rate (float) - basal insulin rate (administered every 5min) in units
    #    sim_duration (int) - simulation duration in minutes

    boluses: List[Bolus] = None

    def __init__(
        self,
        init_bg,
        boluses: List[Bolus],
        meals: List[Meal],
        errors: List[float],
        settings,
        params,
        cgm_config,
        sim_duration=24 * 60,
    ):

        # currently assumes that there are not multiple meals/boluses at the same time
        self.boluses = boluses

        self.meals = meals


        self.sim_duration = sim_duration
        self.params = params
        self.errors = errors
        self.init_bg = init_bg
        self.settings = settings
        self.cgm_config = cgm_config

    def get_events(self, time, state_vec):
        return self.get_bolus(time, state_vec)

    def get_bolus(self, time, state_vec):
        for bolus in self.boluses:
            if not bolus.relative and np.isclose(bolus.time, time):
                return bolus
            elif bolus.relative and np.isclose(bolus.time + np.round(get(state_vec, f'meal_{bolus.meal_index}_time')), time):
                return bolus
        return None
        
    def get_meals(self) -> List[Meal]:
        return self.meals
    
    def get_boluses(self) -> List[Meal]:
        return [self.boluses[t] for t in sorted(self.boluses.keys())]
    
    def get_bolus_meal_mapping(self):
        # maps meal index to bolus
        mapping = {}
        for bolus in self.boluses:
            mapping[bolus.meal_index] = bolus
        return mapping
            
    
    def get_total_carb_range(self):
        total_low = sum([meal.carbs[0] for meal in self.get_meals()])
        total_high = sum([meal.carbs[1] for meal in self.get_meals()])
        return (total_low, total_high)

    def get_data(self):
        return ScenarioData(self.init_bg, self.get_meals(), self.get_boluses(), self.errors, self.settings, self.params, self.cgm_config, self.sim_duration)
    
    def get_largest_meal(self):
        return max([meal.carbs[1] for meal in self.get_meals()])

    def get_smallest_meal(self):
        return min([meal.carbs[0] for meal in self.get_meals()])

    def __key(self):
        return freeze((self.init_bg, self.meals, self.boluses, self.errors, self.params, self.sim_duration, self.settings, self.cgm_config))
    
    def __hash__(self):
        return hash(self.__key)
    
    def __repr__(self):
        return f'Scenario{self.get_meals(), self.boluses}'
    
    def __str__(self):
        return self.__repr__()
    
    def scenario_export():
        '''
        TODO: fill this out for writing scenarios to a file
        '''
        pass
    
    def scenario_import():
        '''
        TODO: fill this out for reading scenario from a file
        '''
        pass
            
