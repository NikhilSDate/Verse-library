from verse_model import State
from body_model import BodyModel

state_indices = {variable: i for i, variable in enumerate(State.__annotations__.keys())}
state_variable_names = list(State.__annotations__.keys())
num_discrete_variables = 1
num_continuous_variables = len(state_indices) - num_discrete_variables
num_meals = 3


def set_val(init, field, value):
    init[state_indices[field]] = value


def get_val(init, field):
    return init[state_indices[field]]


def extract_pump_state(init, pump, events=3):
    iob_array, iob, max_duration = pump.get_state()
    set_val(init, "pump_iob", iob)
    for i in range(events):
        set_val(init, f"pump_iob_{i}", iob_array[i][0])
        set_val(init, f"pump_elapsed_{i}", iob_array[i][1])


def print_state(state):
    a = list(state_indices.keys())[:14]
    for var in a:
        print(f"\t{var}: {state[state_indices[var]]}")


def get_bg(state):
    glucose = state[state_indices["Gp"]] / BodyModel.Vg
    return glucose


###################################
#           MISC UTILS
###################################
def units_to_pmol_per_kg(units):
    # use factor 1 milliunit = 6 pmol
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6501531/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2769591/pdf/dst-01-0323.pdf says insulin infusion rate is pmol/kg/min
    # but it looks like the matlab model uses 1 milliunit = 6.9444 pmol
    return units * 6944.4 / BodyModel.BW
