from typing import List, Union, Tuple
from enum import Enum
import os
from dataclasses import dataclass

class BolusType(str, Enum):
    Simple = 'Simple'
    Extended = 'Extended'

# TODO should we ever model time as a float?

@dataclass(eq=True, frozen=True)
class ExtendedBolusConfig:
    def __init__(self, deliver_now_perc: float, duration: float):
        self.deliver_now_perc: float = deliver_now_perc
        self.duration: float = duration

@dataclass(eq=True, frozen=True)
class Bolus:
    time: int
    carbs: Union[int, Tuple[int, int]]
    type: BolusType
    config: ExtendedBolusConfig

@dataclass(eq=True, frozen=True)
class Meal:
    time: int
    carbs:  Union[int, Tuple[int, int]]
    
    

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
        basal_rate: float,
        boluses: List[Bolus],
        meals: List[Meal],
        iob=0,
        rule="simple",
        sim_duration=24 * 60,
        log_dir='results/logs'
    ):

        self.boluses = {}
        self.meals = {}

        # currently assumes that there are not multiple meals/boluses at the same time
        for bolus in boluses:
            self.boluses[bolus.time] = bolus

        for meal in meals:
            if meal.carbs > 0:
                self.meals[meal.time] = meal

        self.iob = iob
        self.sim_duration = sim_duration

    def get_events(self, time):
        bolus = self.get_bolus(time)
        meal = self.get_meal(time)
        return bolus, meal

    def get_bolus(self, time):
        if time in self.boluses:
            return self.boluses[time]
        return None

    def get_meal(self, time):
        if time in self.meals:
            return self.meals[time]
        return None
    
    def get_meals(self):
        return [self.meals[t] for t in sorted(self.meals.keys())]
