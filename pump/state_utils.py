from verse_model import State

state_indices = {variable: i for i, variable in enumerate(State.__annotations__.keys())}
state_variable_names = list(State.__annotations__.keys())
num_discrete_variables = 1
num_continuous_variables = len(state_indices) - num_discrete_variables
num_meals = 10


def state_set(state_vec, field, value):
    state_vec[state_indices[field]] = value


def state_get(state_vec, field):
    return state_vec[state_indices[field]]
