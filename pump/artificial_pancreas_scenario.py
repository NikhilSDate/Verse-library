from typing import List, Union, Tuple
from enum import Enum
import os

BolusType = Enum("BolusType", ["Simple", "Extended"])

# TODO should we ever model time as a float?


class ExtendedBolusConfig:
    def __init__(self, deliver_now_perc: float, duration: float):
        self.deliver_now_perc: float = deliver_now_perc
        self.duration: float = duration


class Bolus:
    def __init__(self, time: int, carbs: Union[int, Tuple[int, int]], bolus_type: BolusType, config: ExtendedBolusConfig):
        self.time = time
        self.carbs = carbs
        self.type = bolus_type
        self.config = config


class Meal:

    def __init__(self, time: int, carbs: Union[int, Tuple[int, int]]):
        self.time = time  # minutes
        self.carbs = carbs  # grams

class Logger:
    def __init__(self, logging_dir):
        self.dir = logging_dir
        self.sim_idx = 0
        self.current_dose_file = None
        self.current_output_file = None
        
    def start_sim(self):
        if self.current_dose_file:
            self.current_dose_file.close()
        if self.current_output_file:
            self.current_output_file.close()
        dose_path = os.path.join(self.dir, f'sim_{self.sim_idx}_dose.txt')
        output_path = os.path.join(self.dir, f'sim_{self.sim_idx}_output.txt')
        self.current_dose_file = open(dose_path)
        self.current_output_file = open(output_path)
        self.sim_idx += 1

    def write_dose(self, time, dose):
        if self.current_dose_file is None:
            print("Can't log without starting a simulation")
        self.current_dose_file.write(f'')
    
    def __del__(self):
        self.dose_file.close()
        self.output_file.close()


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
