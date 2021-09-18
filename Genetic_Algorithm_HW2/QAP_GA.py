# Enconding should be a matching diagram. The movement operator is a swap in one of the two sets and keep the
# rest the same.

# %%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# %%
def objective_function(distance, flow, departments_set_square, department_assignment):
    z = np.sum(
        [flow[i, j] * distance[department_assignment[i], department_assignment[j]] for i, j in departments_set_square])

    return z


# %%
# Import data
df_distance = pd.read_excel('Genetic_Algorithm_HW2/data_qap.xlsx', sheet_name='Distance', header=None)
# df_distance = pd.read_excel('data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('Genetic_Algorithm_HW2/data_qap.xlsx', sheet_name='Flow', header=None)
# df_flow = pd.read_excel('data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

n_departments = len(distance_matrix)
# %%
# Sets
arcs = np.array([(i, j) for i in range(n_departments) for j in range(n_departments)])

# %% Info Experiments

# np.random.seed(4) # Seed
# pbb_crossover = 0.95
# pbb_mutation = 0.4
# population_size = 100
# max_generation = 1000
# remain_solutions = 2

np.random.seed(4)  # Seed
pbb_crossover = 0.95
pbb_mutation = 0.4
population_size = 200
max_generation = 1000
remain_solutions = 2
# %% Create population

departments_set = np.array([i for i in range(n_departments)])
location_set = np.array([i for i in range(n_departments)])

# Generation initial solutions
population = {}
for creation_sol_iterator in range(population_size):
    random_solution = shuffle(location_set)
    population[creation_sol_iterator] = np.array([random_solution[i] for i in range(n_departments)])

# np.array([shuffle(location_set) for i in range(10)])[1] using array
# Compute fitness for the initial population
population_fitness = {i: objective_function(distance_matrix, flow_matrix, arcs, population[i]) for i in
                      range(population_size)}

# %%

for generation_iterator in range(max_generation):
    # print(generation_iterator)
    children = {}
    children_counter = 0
    for parents in range(int(population_size / 2)):

        # Tournament
        # print('torunament')
        candidates_tourn = np.random.choice([i for i in range(population_size)], size=2, replace=False)
        index_candidate = np.argmin(
            np.array([population_fitness[candidates_tourn[0]], population_fitness[candidates_tourn[1]]]))

        first_parent_index = candidates_tourn[index_candidate]
        second_parent_index = np.random.choice([i for i in range(population_size) if i != first_parent_index])

        # Crossover
        # One point crossover
        # print('crossover')
        if np.random.rand() <= pbb_crossover:

            # Selecting crossover point
            point_selection_crossover = np.random.randint(1, len(location_set) - 1, 1)[0]

            # Creating first half offsprings
            offspring_first = population[first_parent_index][0:point_selection_crossover].copy()
            second_half_first_parent = population[first_parent_index][point_selection_crossover:].copy()

            offspring_second = population[second_parent_index][0:point_selection_crossover].copy()
            second_half_second_parent = population[second_parent_index][point_selection_crossover:].copy()

            # Creating the first offspring
            second_half_second_parent = second_half_second_parent[~np.in1d(second_half_second_parent, offspring_first)]
            offspring_first = np.concatenate([offspring_first, second_half_second_parent])

            missing_in_first_offspring = population[first_parent_index][
                ~np.in1d(population[first_parent_index], offspring_first)]
            offspring_first = np.concatenate([offspring_first, missing_in_first_offspring])

            # Creating the second offspring
            second_half_first_parent = second_half_first_parent[~np.in1d(second_half_first_parent, offspring_second)]
            offspring_second = np.concatenate([offspring_second, second_half_first_parent])

            missing_in_second_offspring = population[second_parent_index][
                ~np.in1d(population[second_parent_index], offspring_second)]
            offspring_second = np.concatenate([offspring_second, missing_in_second_offspring])

        else:
            offspring_first, offspring_second = population[first_parent_index].copy(), population[
                second_parent_index].copy()

        # Mutation
        if np.random.rand() <= pbb_mutation:

            index_swap = np.random.randint(-14, 1, 1)[0]
            direction_swap = np.random.choice([-1, 1], 1)[0]

            offspring_first[index_swap], offspring_first[index_swap + direction_swap] = offspring_first[
                                                                                            index_swap + direction_swap], \
                                                                                        offspring_first[index_swap]

        if np.random.rand() <= pbb_mutation:
            index_swap = np.random.randint(-14, 1, 1)[0]
            direction_swap = np.random.choice([-1, 1], 1)[0]

            offspring_second[index_swap], offspring_second[index_swap + direction_swap] = offspring_second[
                                                                                              index_swap + direction_swap], \
                                                                                          offspring_second[index_swap]

        # %%
        # print('replacement')
        children[children_counter] = offspring_first.copy()
        children_counter += 1
        children[children_counter] = offspring_second.copy()
        children_counter += 1

    # %% Elitism
    children_fitness = {i: objective_function(distance_matrix, flow_matrix, arcs, children[i]) for i in
                        range(len(children))}

    new_generation_children = sorted(children_fitness, key=children_fitness.get, reverse=False)[
                              :population_size - remain_solutions]
    new_generation_parent = sorted(population_fitness, key=population_fitness.get, reverse=False)[:remain_solutions]

    tuple_prev_parents = [([i for i in range(len(new_generation_parent))][j], new_generation_parent[j]) for j in
                          range(remain_solutions)]
    tuple_new_chi = [([i for i in range(remain_solutions, population_size)][j], new_generation_children[j]) for j in
                     range(population_size - remain_solutions)]

    population = {i: population[j].copy() for i, j in tuple_prev_parents}
    for i, j in tuple_new_chi:
        population[i] = children[j].copy()

    # population = children.copy()

    population_fitness = {i: objective_function(distance_matrix, flow_matrix, arcs, population[i]) for i in
                          range(len(population))}
    print(min(population_fitness.values()))
