# %%
import numpy as np
import pandas as pd


# %%
def eggholder(x, y):
    z = -(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2 + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return z


pop_size = 10

x_lb = -512
x_ub = 512
y_lb = -512
y_ub = 512

v_max = x_ub / 40
v_min = x_lb / 40

np.random.seed(1)
velocity_den = 20

phi_1 = 2
phi_2 = 2

iterations_counter = 0
number_iterations = 100
counter_phi = 0
max_phi_iteration = 20

# %% Saving vector

best_objective_save = []
fitness_save = []
df_best_objective = pd.DataFrame()

# %% Initialize particles

p_vector = np.array([[np.random.uniform(x_lb, x_ub), np.random.uniform(y_lb, y_ub)] for i in range(pop_size)])
p_fitness = np.zeros(pop_size)
p_velocity = np.array([[np.random.uniform(v_min, v_max), np.random.uniform(v_min, v_max)] for i in range(pop_size)])
p_position = p_vector.copy()
p_position_fitness = p_fitness.copy()
# %% Start iterations

while iterations_counter <= number_iterations:
    for i in range(pop_size):
        p_fitness[i] = eggholder(p_vector[i][0], p_vector[i][1])
        fitness_save.append(p_fitness)
        if p_fitness[i] <= eggholder(p_position[i][0], p_position[i][1]):
            p_position[i] = p_vector[i].copy()
            p_position_fitness[i] = eggholder(p_position[i][0], p_position[i][1])

    best_particle_index = np.argmin(p_position_fitness)
    best_particle = p_position[best_particle_index].copy()

    for i in range(pop_size):

        p_velocity_cand_x = 0
        p_velocity_cand_y = 0

        p_velocity_cand_x = p_velocity[i][0] + np.random.uniform(0, phi_1) * (
                p_position[i][0] - p_vector[i][0]) + np.random.uniform(0, phi_2) * (best_particle[0] - p_vector[i][0])

        while p_velocity_cand_x <= v_min or p_velocity_cand_x >= v_max:
            counter_phi += 1
            p_velocity_cand_x = p_velocity[i][0] + np.random.uniform(0, phi_1) * (
                    p_position[i][0] - p_vector[i][0]) + np.random.uniform(0, phi_2) * (
                                            best_particle[0] - p_vector[i][0])

            if counter_phi >= max_phi_iteration:
                phi_1 = phi_1 * 0.9
                phi_2 = phi_2 * 0.9

        p_velocity[i][0] = p_velocity_cand_x.copy()

        p_velocity_cand_y = p_velocity[i][1] + np.random.uniform(0, phi_1) * (
                p_position[i][1] - p_vector[i][1]) + np.random.uniform(0, phi_2) * (best_particle[1] - p_vector[i][1])

        while p_velocity_cand_y <= v_min or p_velocity_cand_y >= v_max:
            counter_phi += 1
            p_velocity_cand_y = p_velocity[i][1] + np.random.uniform(0, phi_1) * (
                    p_position[i][1] - p_vector[i][1]) + np.random.uniform(0, phi_2) * (
                                            best_particle[1] - p_vector[i][1])

            if counter_phi >= max_phi_iteration:
                phi_1 = phi_1 * 0.9
                phi_2 = phi_2 * 0.9

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

        # print(p_vector[i][0], p_vector[i][1])

    print(eggholder(best_particle[0], best_particle[1]))
    best_objective_save.append(eggholder(best_particle[0], best_particle[1]))
    iterations_counter += 1

