# Enconding should be a matching diagram. The movement operator is a swap in one of the two sets and keep the
# rest the same.

# %%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# %%
def z_function(x, y):
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.exp(1)
    return z

def correction_factor(l_bound, u_bound, n_bits):
    return (u_bound-l_bound) / ((2 ** n_bits) - 1)


# %%

x_lower_bound = -32
x_upper_bound = 32
y_lower_bound = -32
y_upper_bound = 32

n_bits = 10

population_size = 50
n_variables = 2

np.random.seed(4)  # Seed
pbb_crossover = 0.95
pbb_mutation = 0.35
max_generation = 1000
remain_solutions = 3

np.random.seed(0)
decoding = 2**np.arange(n_bits-1, -1, -1)
x_correction = correction_factor(x_lower_bound, x_upper_bound, n_bits)
y_correction = correction_factor(y_lower_bound, y_upper_bound, n_bits)

# %% Initial population

encoding = np.reshape(np.random.choice([0, 1], size=population_size * n_variables * n_bits), (population_size, n_variables, n_bits)) # shape: number solutions, number variables, number of bits

population = {i : encoding[i] for i in range(population_size)}
population_fitness = {i : z_function(sum(decoding * encoding[i][0]) * x_correction + x_lower_bound, sum(decoding * encoding[i][1]) * y_correction + y_lower_bound) for i in range(population_size)}
print(min(population_fitness.values()))

# %%

for generation_iterator in range(max_generation):

    children = {}
    children_counter = 0
    for parents in range(int(population_size / 2)):

        # Tournament
        # print('torunament')
        candidates_tourn = np.random.choice([i for i in range(population_size)], size=2, replace=False)
        index_candidate = np.argmin(np.array([population_fitness[candidates_tourn[0]], population_fitness[candidates_tourn[1]]]))

        first_parent_index = candidates_tourn[index_candidate]
        second_parent_index = np.random.choice([i for i in range(population_size) if i != first_parent_index])

        # Crossover
        # One point crossover
        if np.random.rand() <= pbb_crossover:

            # Selecting crossover point
            point_selection_crossover = np.random.randint(1, n_bits - 1, 1)[0]

            # Creating first half offsprings
            offspring_first = np.array([np.concatenate([population[first_parent_index][0][0:point_selection_crossover].copy(), population[second_parent_index][0][point_selection_crossover:].copy()]),
                                          np.concatenate([population[first_parent_index][1][0:point_selection_crossover].copy(), population[second_parent_index][1][point_selection_crossover:].copy()])])

            offspring_second = np.array([np.concatenate([population[second_parent_index][0][0:point_selection_crossover].copy(), population[first_parent_index][0][point_selection_crossover:].copy()]),
                                          np.concatenate([population[second_parent_index][1][0:point_selection_crossover].copy(), population[first_parent_index][1][point_selection_crossover:].copy()])])

        else:
            offspring_first, offspring_second = population[first_parent_index].copy(), population[second_parent_index].copy()

        # Mutation
        random_number_evaluation = np.random.random(size=offspring_first.shape)
        mask_mutation = random_number_evaluation < pbb_mutation
        offspring_first_m = offspring_first.copy()
        offspring_first_m[mask_mutation] = np.where((offspring_first_m[mask_mutation] == 0) | (offspring_first_m[mask_mutation] == 1), offspring_first_m[mask_mutation] ^ 1, offspring_first_m[mask_mutation])

        random_number_evaluation = np.random.random(size=offspring_second.shape)
        mask_mutation = random_number_evaluation < pbb_mutation
        offspring_second_m = offspring_second.copy()
        offspring_second_m[mask_mutation] = np.where((offspring_second_m[mask_mutation] == 0) | (offspring_second_m[mask_mutation] == 1), offspring_second_m[mask_mutation] ^ 1, offspring_second_m[mask_mutation])

        # Adding new children
        children[children_counter] = offspring_first_m.copy()
        children_counter += 1
        children[children_counter] = offspring_second_m.copy()
        children_counter += 1

    # Elitism
    children_fitness = {i: z_function(sum(decoding * children[i][0]) * x_correction + x_lower_bound, sum(decoding * children[i][1]) * y_correction + y_lower_bound) for i in
                        range(len(children))}

    new_generation_children = sorted(children_fitness, key=children_fitness.get, reverse=False)[
                              :population_size - remain_solutions]
    new_generation_parent = sorted(population_fitness, key=population_fitness.get, reverse=False)[:remain_solutions]

    tuple_prev_parents = [([i for i in range(len(new_generation_parent))][j], new_generation_parent[j]) for j in
                          range(remain_solutions)]
    tuple_new_chi = [([i for i in range(remain_solutions, population_size)][j], new_generation_children[j]) for j in
                     range(population_size - remain_solutions)]
    #pr0int('replacement')
    population = {i: population[j].copy() for i, j in tuple_prev_parents}
    for i, j in tuple_new_chi:
        population[i] = children[j].copy()

    # population = children.copy()

    population_fitness = {i: z_function(sum(decoding * population[i][0]) * x_correction + x_lower_bound, sum(decoding * population[i][1]) * y_correction + y_lower_bound) for i in range(population_size)}
    print(min(population_fitness.values()))