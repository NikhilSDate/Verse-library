import os, sys
from dotenv import load_dotenv

load_dotenv()
EMULATOR_PATH = os.environ["EMULATOR_PATH"]

sys.path.insert(1, EMULATOR_PATH)
from pump_wrapper import Pump

from verse_model import State
from body_model import BodyModel
from artificial_pancreas_scenario import *


class InsulinPumpEvent:

    def __init__(self, iob: float, time_elapsed: float):
        self.iob = iob
        self.time_elapsed = time_elapsed


class InsulinPumpModel:

    def __init__(self, sim_scenario):

        self.current_iob = sim_scenario.iob
        self.basal_rate = sim_scenario.basal_rate

        pump = Pump()
        self.pump_emulator = pump

    def send_bolus_command(self, bg, bolus: Bolus):
        if bolus.type == BolusType.Simple:
            # TODO why is this +30?
            dose = self.pump_emulator.dose_simple(bg, bolus.carbs)
            return dose
        else:
            # TODO implement extended bolus
            # glucose = get_visible(init)[0]
            # carbs = time_to_carbs[current_time]
            # pump.dose_extended(glucose, carbs, 50, 120)
            # print('dosing')
            raise NotImplementedError()


###################################
#           MISC UTILS
###################################
def units_to_pmol_per_kg(units):
    # use factor 1 milliunit = 6 pmol
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501531/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769591/pdf/dst-01-0323.pdf says insulin infusion rate is pmol/kg/min
    # but it looks like the matlab model uses 1 milliunit = 6.9444 pmol
    return units * 6944.4 / BodyModel.BW  # TODO this should be an instance of body model
