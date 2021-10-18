#%%
import numpy as np
import pandas as pd
#%%
def eggholder(x, y):
    z = -(y + 47) * np.sin(np.sqrt(np.abs(y + x/2 + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return z

def rnd_1_bin(n_dim, target, parents):
    sel_parents = np.random.default_rng().choice(parents, size=3, replace=False)
    while target in sel_parents:
        sel_parents = np.random.default_rng().choice(parents, size=3, replace=False)
    mutant = np.array([sel_parents[0, dim] + f * (sel_parents[1, dim] - sel_parents[2, dim]) for dim in range(n_dim)])
    return mutant

def best_1_bin(n_dim, target, parents):
    best_index = np.where(parents == np.amin([parents[i][2] for i in range(pop_size)]))[0][0]
    best = population[best_index]
    sel_parents = np.random.default_rng().choice(parents, size=2, replace=False)
    while target in sel_parents:
        sel_parents = np.random.default_rng().choice(parents, size=2, replace=False)
    mutant = np.array([best[dim] + f * (sel_parents[0, dim] - sel_parents[1, dim]) for dim in range(n_dim)])
    return mutant

x_lb = -512
x_ub = 512
y_lb = -512
y_ub = 512

#%%
pop_size = 40
#f = 0.6
cr = 0.9
max_generations = 100
n_dim = 2

#%% Experiment loop:
for seed_it in [0, 1, 2, 3, 4]:
#%% Saving list
    best_solutions_generation = []

    #%%
    np.random.seed(seed_it)
    population = np.array([[np.random.uniform(x_lb, x_ub), np.random.uniform(y_lb, y_ub)] for i in range(pop_size)])
    population_z = np.array([eggholder(population[i][0], population[i][1]) for i in range(pop_size)])
    population_z = np.reshape(population_z, (pop_size, 1))
    population = np.append(population, population_z, 1)
    parents = population.copy()

    best_solution_index = np.argmin([population[i][2] for i in range(pop_size)])

    best_solutions_generation.append(population[best_solution_index])
    print(population[best_solution_index])
    #%%
    np.random.seed(0)
    for gen_it in range(max_generations):
        f = np.random.uniform(0.5, 1)
        for target_vector in population:

            mutant_vector = rnd_1_bin(n_dim, target_vector, parents)
            #mutant_vector = best_1_bin(n_dim, target_vector, parents)
            j_rand = np.random.randint(1, n_dim, size = 1)
            trial_vector = np.zeros((3))
            for j in range(n_dim):
                if (np.random.rand() <= cr) or (j == j_rand):
                    trial_vector[j] = mutant_vector[j]
                else:
                    trial_vector[j] = target_vector[j]
            trial_vector[2] = eggholder(trial_vector[0], trial_vector[1])

            if (target_vector[2] > trial_vector[2]) and (x_lb <= trial_vector[0] < x_ub) and (y_lb <= trial_vector[1] < y_ub):
                parents[np.where(population == target_vector)[0][0]] = trial_vector

        population = parents
        best_solutions_generation.append(population[best_solution_index].copy())

    df = pd.DataFrame(best_solutions_generation, columns = ['x', 'y', 'z'])
    df.to_excel(f'DE_HW4/de_rand_objective_seed{seed_it}_fdynamic.xlsx', index=False)


