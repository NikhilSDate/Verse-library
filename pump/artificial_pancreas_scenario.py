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

@dataclass(eq=True, frozen=True)
class Meal:
    time: int
    carbs: Union[int, Tuple[int, int]]
    TauM: float

    
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
        sim_duration=24 * 60,
    ):

        self.boluses: Dict[int, Bolus] = {}
        self.meals = {}

        # currently assumes that there are not multiple meals/boluses at the same time
        for bolus in boluses:
            self.boluses[bolus.time] = bolus

        for meal in meals:
            if (np.ndim(meal.carbs) == 0 and meal.carbs > 0) or (meal.carbs[1] > 0):       
                self.meals[meal.time] = meal

        self.sim_duration = sim_duration
        self.params = params
        self.errors = errors
        self.init_bg = init_bg
        self.settings = settings
        self.params = params

    def get_events(self, time, state_vec):
        return self.get_bolus(time)

    def get_bolus(self, time):
        if time in self.boluses:
            return self.boluses[time]
        return None

    def get_meal(self, time):
        if time in self.meals:
            return self.meals[time]
        return None
    
    def get_meals(self) -> List[Meal]:
        return [self.meals[t] for t in sorted(self.meals.keys())]
    
    def get_bolus_meal_mapping(self):
        # maps meal index to bolus
        mapping = {}
        for bolus in self.boluses.values():
            mapping[bolus.meal_index] = bolus
        return mapping
            
    
    def __repr__(self):
        return f'Scenario{self.get_meals(), self.boluses.values()}'
    
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
            
