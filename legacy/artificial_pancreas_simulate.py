# dump of some old functions from artificial_pancreas_simulate.py

def verify_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):

    agent = ArtificialPancreasAgent("pump", file_name="verse_model.py")
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (PumpMode.default,))
    traces = scenario.verify(duration, time_step)
    end_low = traces.root.trace["pump"][-2]
    end_high = traces.root.trace["pump"][-1]
    return [end_low[1:], end_high[1:]], traces


def simulate_bolus(init, carbs_low, carbs_high, duration=360, time_step=1):

    agent = ArtificialPancreasAgent("pump", file_name="verse_model.py")
    scenario = SimulationScenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (PumpMode.default,))
    traces = scenario.simulate(duration, time_step)
    end = traces.root.trace["pump"][-1]
    return [end[1:], end[1:]], traces


def link_nodes(node1, node2):
    agent = list(node1.trace.keys())[0]
    trace1_end = node1.trace[agent][-1][0]
    trace2_len = len(node2.trace[agent])
    for i in range(trace2_len):
        node2.trace[agent][i][0] += trace1_end
    node2.height = node1.height + 1
    node2.id = node1.id + 1
    node1.child.append(node2)


# boluses is time, [carbs_low, carbs_high]
def verify_boluses(init, boluses, duration=120):
    result1, tree1 = verify_bolus(init, boluses[0][1][0], boluses[0][1][1], duration)
    prev_result, prev_tree = result1, tree1
    for i in range(1, len(boluses)):
        result, tree = verify_bolus(
            copy.deepcopy(prev_result), boluses[i][1][0], boluses[i][1][1], duration
        )
        link_nodes(prev_tree.root, tree.root)
        prev_result, prev_tree = result, tree
    return result1, tree1

def generate_all_three_meal_traces(
    init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs, trace_directory=TRACES_PATH
):

    all_combinations = np.array(
        np.meshgrid(init_bg, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs)
    ).T.reshape(-1, 5)
    existing_files = os.listdir(trace_directory)
    for i in tqdm(range(len(all_combinations))):
        combination = all_combinations[i]
        tqdm.write(f"Simulating with init state {combination}")
        bg, br, bc, lc, dc = combination
        filename = f"trace_{bg}_{br}_{bc}_{lc}_{dc}.csv"
        if filename in existing_files:
            continue
        traces = simulate_three_meal_scenario(bg, br, bc, lc, dc)
        save_traces(traces, os.path.join(trace_directory, filename))


def verify_three_meal_scenario(init_bg, BW, basal_rate, breakfast_carbs, lunch_carbs, dinner_carbs):
    meals_low = [(breakfast_carbs[0], 0), (lunch_carbs[0], 240), (dinner_carbs[0], 660)]
    meals_high = [(breakfast_carbs[1], 0), (lunch_carbs[1], 240), (dinner_carbs[1], 660)]
    init_state_low = get_init_state(init_bg[0], BW, meals_low)
    init_state_high = get_init_state(init_bg[1], BW, meals_high)
    init = [init_state_low, init_state_high]
    simulation_scenario = SimulationScenario(meals_low, basal_rate)
    agent = ArtificialPancreasAgent(
        "pump", simulation_scenario=simulation_scenario, file_name=PUMP_PATH + "verse_model.py"
    )
    scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))
    scenario.add_agent(agent)
    scenario.set_init_single("pump", init, (PumpMode.default,))
    duration = simulation_scenario.simulation_duration
    time_step = 1
    traces = scenario.verify(duration, time_step)
    return traces
