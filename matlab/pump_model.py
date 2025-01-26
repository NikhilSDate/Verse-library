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

    def __init__(self, sim_scenario, basal_iq=False, settings=None):

        self.current_iob = sim_scenario.iob
        self.basal_rate = sim_scenario.basal_rate    
        self.settings = settings
        self.basal_iq = basal_iq        
        self.pump_emulator = self.get_pump(basal_iq, settings)
    
    def reset_pump(self):
        self.pump_emulator = self.get_pump(self.basal_iq, self.settings)
    
    def get_pump(self, basal_iq, settings):
        pump = Pump(basal_iq=basal_iq)
        if settings is not None:
            pump.set_settings(carb_ratio=settings['carb_ratio'], correction_factor=settings['correction_factor'], target_bg=settings['target_bg'], max_bolus=settings['max_bolus'], insulin_duration=settings['insulin_duration'], basal_rate=settings['basal_rate'])
        return pump

    def send_bolus_command(self, bg, bolus: Bolus):
        if bolus.type == BolusType.Simple:
            # TODO why is this +30?
            self.pump_emulator.dose_simple(bg, bolus.carbs)
        else:
            self.pump_emulator.dose_extended(bg, bolus.carbs, bolus.config.deliver_now_perc, bolus.config.duration)
            
    def extract_state(self) -> Tuple[float]:
        state = self.pump_emulator.get_state()
        return state
    
    def get_init_state(self):
        return [0]


###################################
#           MISC UTILS
###################################
def units_to_pmol_per_kg(units):
    # use factor 1 milliunit = 6 pmol
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501531/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769591/pdf/dst-01-0323.pdf says insulin infusion rate is pmol/kg/min
    # but it looks like the matlab model uses 1 milliunit = 6.9444 pmol
    return units * 6944.4 / BodyModel.BW  # TODO this should be an instance of body model
