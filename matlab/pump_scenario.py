from typing import List
from enum import Enum

BolusType = Enum("BolusType", ["Simple", "Extended"])


class ExtendedBolusConfig:
    def __init__(self, deliver_now_percentage: float, duration: float):
        self.deliver_now_percentage: float = deliver_now_percentage
        self.duration: float = duration


class Bolus:
    def __init__(self, time, carbs, bolus_type: BolusType, config: ExtendedBolusConfig):
        self.time = time
        self.carbs = carbs
        self.type: BolusType = bolus_type
        self.config: ExtendedBolusConfig = config


class SimulationScenario:
#    def __init__(self, meals, basal_rate, rule="simple", simulation_duration=24):
    def __init__(self, meals, basal_rate, rule="simple", simulation_duration=24 * 60):
        # eventually we can define some logic for when we should request a bolus
        # meals is array of (time, carbs)
        self.boluses: List[Bolus] = []
        for carbs, time in meals:
            if carbs > 0:
                self.boluses.append(Bolus(time, carbs, BolusType.Simple, None))
        self.basal_rate = basal_rate
        self.simulation_duration = simulation_duration

    def get_bolus(self, time):
        for bolus in self.boluses:
            if bolus.time == time:
                return bolus
        return None
