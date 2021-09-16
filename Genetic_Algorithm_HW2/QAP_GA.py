# Enconding should be a matching diagram. The movement operator is a swap in one of the two sets and keep the
# rest the same.

#%%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#%%
def objective_function(distance, flow, departments_set_square, department_assignment):

    z = np.sum([flow[i, j] * distance[department_assignment[i], department_assignment[j]] for i, j in departments_set_square])

    return z

#%%
def swap_movement(first_set, second_set):

    random_swap = np.random.choice(second_set, 1, replace=False)
    index_one = np.where(second_set == random_swap[0])
    index_two = index_one - np.random.choice(np.array([1, 2, 3]), 1, replace=False)
    second_set[index_one], second_set[index_two] = second_set[index_two], second_set[index_one]
    new_assignment = {first_set[i]: second_set[i] for i in first_set}

    return new_assignment

#%%
# Import data
df_distance = pd.read_excel('Genetic_Algorithm_HW2/data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('Genetic_Algorithm_HW2/data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

n_departments = len(distance_matrix)
#%%
# Sets
arcs = np.array([(i, j) for i in range(n_departments) for j in range(n_departments)])

#%% Info Experiments

np.random.seed(0) # Seed
pbb_crossover = 0.9
pbb_mutation = 0.3
population_size = 50
max_generation = 10000

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
population_fitness = {i: objective_function(distance_matrix, flow_matrix, arcs, population[i]) for i in range(population_size)}

#%%

for generation_iterator in range(max_generation):
    #print(generation_iterator)
    children = {}
    children_counter = 0
    for parents in range(int(population_size/2)):

        # Tournament
        #print('torunament')
        candidates_tourn = np.random.randint(0, population_size, size=2)
        index_candidate = np.argmax(np.array([population_fitness[candidates_tourn[0]], population_fitness[candidates_tourn[1]]]))

        first_parent_index = candidates_tourn[index_candidate]
        second_parent_index = np.random.choice([i for i in range(10) if i !=first_parent_index])

        # Crossover
        # One point crossover
        #print('crossover')
        if np.random.rand() <= pbb_crossover:

            # Selecting crossover point
            point_selection_crossover = np.random.randint(1, len(location_set)-1, 1)[0]

            # Creating first half offsprings
            offspring_first = population[first_parent_index][0:point_selection_crossover]
            second_half_first_parent = population[first_parent_index][point_selection_crossover:]

            offspring_second = population[second_parent_index][0:point_selection_crossover]
            second_half_second_parent = population[second_parent_index][point_selection_crossover:]

            # Creating the first offspring
            mask_1 = np.setdiff1d(second_half_second_parent, offspring_first)
            second_half_second_parent = second_half_second_parent[~np.in1d(second_half_second_parent, offspring_first)]
            offspring_first = np.concatenate([offspring_first, second_half_second_parent])

            missing_in_first_offspring = population[first_parent_index][~np.in1d(population[first_parent_index], offspring_first)]
            offspring_first = np.concatenate([offspring_first, missing_in_first_offspring])

            # Creating the second offspring
            second_half_first_parent = second_half_first_parent[~np.in1d(second_half_first_parent, offspring_second)]
            offspring_second = np.concatenate([offspring_second, second_half_first_parent])

            missing_in_second_offspring = population[second_parent_index][~np.in1d(population[second_parent_index], offspring_second)]
            offspring_second = np.concatenate([offspring_second, missing_in_second_offspring])

        else:
            offspring_first, offspring_second = population[first_parent_index], population[second_parent_index]

        # Mutation
        #print('mutation')
        if np.random.rand() <= pbb_mutation:
            index_swap = np.random.randint(-13, 2, 1)[0]
            direction_swap = np.random.choice([-2, 2, -1, 1], 1)[0]

            offspring_first[index_swap], offspring_first[index_swap + direction_swap] = offspring_first[index_swap + direction_swap], \
                                                                                        offspring_first[index_swap]

            index_swap = np.random.randint(-13, 2, 1)[0]
            direction_swap = np.random.choice([-2, 2, -1, 1], 1)[0]

            offspring_second[index_swap], offspring_second[index_swap + direction_swap] = offspring_second[index_swap + direction_swap], \
                                                                                        offspring_second[index_swap]

# %%
        #print('replacement')
        children[children_counter] = offspring_first
        children_counter += 1
        children[children_counter] = offspring_second
        children_counter += 1

    children_fitness = {i: objective_function(distance_matrix, flow_matrix, arcs, children[i]) for i in range(len(children))}

    new_generation_children = sorted(children_fitness, key=children_fitness.get, reverse=False)[:25]
    new_generation_parent = sorted(population_fitness, key=population_fitness.get, reverse=False)[:25]

    tuple_new_gen = [([i for i in range(len(new_generation_parent))][j], new_generation_parent[j]) for j in range(25)]
    tuple_new_chi = [([i for i in range(25, 50)][j], new_generation_children[j]) for j in range(0, 25)]

    population = {i: population[j] for i,j in tuple_new_gen}
    for i, j in tuple_new_chi:
        population[i] = children[j]

    #print('new fitness')
    #population = children
    population_fitness = {i: objective_function(distance_matrix, flow_matrix, arcs, population[i]) for i in range(len(population))}
    print(min(population_fitness.values()))
