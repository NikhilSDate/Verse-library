from real_pump_matlab_model import State

state_indices = {variable: i for i, variable in enumerate(State.__annotations__.keys())}
num_discrete_variables = 1
num_continuous_variables = len(state_indices) - num_discrete_variables
num_meals = 3


def set(init, field, value):
    init[state_indices[field]] = value


def get(init, field):
    return init[state_indices[field]]
