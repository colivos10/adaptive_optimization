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

np.random.seed(1)
velocity_den = 100

phi_1 = 2
phi_2 = 2

iterations_counter = 0
number_iterations = 50

# %% Initialize particles

p_vector = np.array([[np.random.uniform(x_lb, x_ub), np.random.uniform(y_lb, y_ub)] for i in range(pop_size)])
p_fitness = np.zeros(pop_size)
p_velocity = np.array([[np.random.uniform(x_lb / velocity_den, x_ub / velocity_den),
                        np.random.uniform(y_lb / velocity_den, y_ub / velocity_den)] for i in range(pop_size)])
p_position = p_vector.copy()
p_position_fitness = p_fitness.copy()
# %% Start iterations

while iterations_counter <= number_iterations:
    for i in range(pop_size):
        p_fitness[i] = eggholder(p_vector[i][0], p_vector[i][1])
        if p_fitness[i] <= eggholder(p_position[i][0], p_position[i][1]):
            p_position[i] = p_vector[i].copy()
            p_position_fitness[i] = eggholder(p_position[i][0], p_position[i][1])

    best_particle_index = np.argmin(p_position_fitness)
    best_particle = p_position[best_particle_index].copy()

    for i in range(pop_size):
        while p_vector[i][0] + p_velocity[i][0] <= x_lb or p_vector[i][0] + p_velocity[i][0] >= x_ub:

            p_velocity[i][0] = p_velocity[i][0] + np.random.uniform(0, phi_1) * (
                        p_position[i][0] - p_vector[i][0]) + phi_2 * np.random.rand() * (best_particle[0] - p_vector[i][0])

        while p_vector[i][1] + p_velocity[i][1] <= y_lb or p_vector[i][1] + p_velocity[i][1] >= y_ub:

            p_velocity[i][1] = p_velocity[i][1] + np.random.uniform(0, phi_2) * (
                    p_position[i][1] - p_vector[i][1]) + phi_2 * np.random.rand() * (best_particle[1] - p_vector[i][1])

        p_vector[i][0] = p_vector[i][0] + p_velocity[i][0]
        p_vector[i][1] = p_vector[i][1] + p_velocity[i][1]

    #print(best_particle)
    print(eggholder(best_particle[0], best_particle[1]))
    iterations_counter += 1