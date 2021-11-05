# %%
import numpy as np
import pandas as pd


# %%
def eggholder(x, y):
    z = -(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2 + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return z

# %%
pop_size = 15

x_lb = -512
x_ub = 512
y_lb = -512
y_ub = 512

v_max = x_ub / 40
v_min = x_lb / 40

phi_1 = 2
phi_2 = 2

# %% Saving vector

df_best_objective = pd.DataFrame()
df_fitness_objectives = pd.DataFrame()

for seed_it in range(10):
    print(seed_it)
    phi_1 = 2
    phi_2 = 2
    w_inertia = 0.9
    iterations_counter = 0
    number_iterations = 100
    counter_phi = 0
    max_phi_iteration = 20
    best_objective_save = []
    fitness_save = []

    np.random.seed(seed_it)

    # %% Initialize particles
    p_vector = np.array([[np.random.uniform(x_lb, x_ub), np.random.uniform(y_lb, y_ub)] for i in range(pop_size)])
    p_fitness = np.zeros(pop_size)
    p_velocity = np.array([[np.random.uniform(v_min, v_max), np.random.uniform(v_min, v_max)] for i in range(pop_size)])
    p_position = p_vector.copy()
    p_position_fitness = p_fitness.copy()
    # %% Start iterations

    while iterations_counter <= number_iterations:

        if iterations_counter % 10 == 0:
            w_inertia = w_inertia * 0.9

        for i in range(pop_size):
            p_fitness[i] = eggholder(p_vector[i][0], p_vector[i][1])
            fitness_save.append(p_fitness[i])

            if p_fitness[i] <= eggholder(p_position[i][0], p_position[i][1]):
                p_position[i] = p_vector[i].copy()
                p_position_fitness[i] = eggholder(p_position[i][0], p_position[i][1])

        best_particle_index = np.argmin(p_position_fitness)
        best_particle = p_position[best_particle_index].copy()

        for i in range(pop_size):

            p_velocity_cand_x = 0
            p_velocity_cand_y = 0

            p_velocity_cand_x = w_inertia * p_velocity[i][0] + np.random.uniform(0, phi_1) * (
                    p_position[i][0] - p_vector[i][0]) + np.random.uniform(0, phi_2) * (best_particle[0] - p_vector[i][0])

            p_velocity[i][0] = p_velocity_cand_x.copy()

            p_velocity_cand_y = w_inertia * p_velocity[i][1] + np.random.uniform(0, phi_1) * (
                    p_position[i][1] - p_vector[i][1]) + np.random.uniform(0, phi_2) * (best_particle[1] - p_vector[i][1])

            p_velocity[i][1] = p_velocity_cand_y.copy()

            p_vector[i][0] = p_vector[i][0] + p_velocity[i][0]
            p_vector[i][1] = p_vector[i][1] + p_velocity[i][1]

            if p_vector[i][0] <= x_lb:
                p_vector[i][0] = max(x_lb, p_vector[i][0])
            elif p_vector[i][0] >= x_ub:
                p_vector[i][0] = min(x_ub, p_vector[i][0])

            if p_vector[i][1] <= y_lb:
                p_vector[i][1] = max(y_lb, p_vector[i][1])
            elif p_vector[i][1] >= x_ub:
                p_vector[i][1] = min(y_ub, p_vector[i][1])

        #print(eggholder(best_particle[0], best_particle[1]))
        best_objective_save.append(eggholder(best_particle[0], best_particle[1]))
        iterations_counter += 1

    df_fitness_objectives[f'Seed {seed_it}'] = fitness_save
    df_best_objective[f'Seed {seed_it}'] = best_objective_save

df_fitness_objectives.to_excel('PSO_HW6/results/fitness_total_v6.xlsx', index = False)
df_best_objective.to_excel('PSO_HW6/results/best_objectives_v6.xlsx', index = False)