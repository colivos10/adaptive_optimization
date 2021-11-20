# %%
import numpy as np
import pandas as pd


# %%
def schwefel(x, y, w, z, alpha, n):
    obj_val = -x * np.sin(np.sqrt(np.abs(x))) + (-y * np.sin(np.sqrt(np.abs(y)))) +\
              (-w * np.sin(np.sqrt(np.abs(w)))) + (-z * np.sin(np.sqrt(np.abs(z)))) + alpha * n
    return obj_val

#schwefel(420.968746, 420.968746, 420.968746, 420.968746, 418.982887, 4)

# %%
alpha = 418.982887
n = 4

pop_size = 100

x_lb = -512
x_ub = 512
y_lb = -512
y_ub = 512
w_lb = -512
w_ub = 512
z_lb = -512
z_ub = 512



# %% Saving vector

for pop_it in [50, 150]:
    df_best_objective = pd.DataFrame()
    df_fitness_objectives = pd.DataFrame()
    pop_size = pop_it
    for seed_it in range(10):

        v_max = x_ub / 40
        v_min = x_lb / 40
        phi_1 = 2.1
        phi_2 = 2.1
        k_constriction = 2 / np.abs(2 - (phi_1 + phi_2) - np.sqrt((phi_1 + phi_2) ** 2 - 4 * (phi_1 + phi_2)))
        iterations_counter = 0
        number_iterations = 100
        counter_phi = 0
        max_phi_iteration = 100
        best_objective_save = []
        fitness_save = []
        w_inertia = 0.9
        np.random.seed(seed_it)

        # %% Initialize particles
        p_vector = np.array([[np.random.uniform(x_lb, x_ub), np.random.uniform(y_lb, y_ub),
                              np.random.uniform(w_lb, w_ub), np.random.uniform(z_lb, z_ub)] for i in range(pop_size)])
        p_fitness = np.zeros(pop_size)
        p_velocity = np.array([[np.random.uniform(v_min, v_max), np.random.uniform(v_min, v_max),
                                np.random.uniform(v_min, v_max), np.random.uniform(v_min, v_max) ] for i in range(pop_size)])
        p_position = p_vector.copy()
        p_position_fitness = p_fitness.copy()
        # %% Start iterations
        while iterations_counter <= number_iterations:

            if iterations_counter % 10 == 0:
                w_inertia = w_inertia * 0.9

            for i in range(pop_size):
                p_fitness[i] = schwefel(p_vector[i][0], p_vector[i][1], p_vector[i][2], p_vector[i][3], alpha, n)
                fitness_save.append(p_fitness[i])
                if p_fitness[i] <= schwefel(p_position[i][0], p_position[i][1], p_position[i][2], p_position[i][3], alpha, n):
                    p_position[i] = p_vector[i].copy()
                    p_position_fitness[i] = schwefel(p_position[i][0], p_position[i][1], p_position[i][2], p_position[i][3],
                                                     alpha, n)

            best_particle_index = np.argmin(p_position_fitness)
            best_particle = p_position[best_particle_index].copy()

            for i in range(pop_size):

                p_velocity_cand_x = 0
                p_velocity_cand_y = 0
                p_velocity_cand_w = 0
                p_velocity_cand_z = 0

                # For x
                p_velocity_cand_x = k_constriction * (w_inertia * p_velocity[i][0] + np.random.uniform(0, phi_1) * (
                        p_position[i][0] - p_vector[i][0]) + np.random.uniform(0, phi_2) * (best_particle[0] - p_vector[i][0]))

                p_velocity[i][0] = p_velocity_cand_x.copy()

                # For y

                p_velocity_cand_y = k_constriction * (w_inertia * p_velocity[i][1] + np.random.uniform(0, phi_1) * (
                        p_position[i][1] - p_vector[i][1]) + np.random.uniform(0, phi_2) * (best_particle[1] - p_vector[i][1]))

                p_velocity[i][1] = p_velocity_cand_y.copy()

                # For w
                p_velocity_cand_w = k_constriction * (w_inertia * p_velocity[i][0] + np.random.uniform(0, phi_1) * (
                        p_position[i][0] - p_vector[i][0]) + np.random.uniform(0, phi_2) * (best_particle[0] - p_vector[i][0]))

                p_velocity[i][2] = p_velocity_cand_w.copy()

                # For z
                p_velocity_cand_z = k_constriction * (w_inertia * p_velocity[i][3] + np.random.uniform(0, phi_1) * (
                        p_position[i][3] - p_vector[i][3]) + np.random.uniform(0, phi_2) * (best_particle[3] - p_vector[i][3]))

                p_velocity[i][3] = p_velocity_cand_z.copy()

                p_vector[i][0] = p_vector[i][0] + p_velocity[i][0]
                p_vector[i][1] = p_vector[i][1] + p_velocity[i][1]
                p_vector[i][2] = p_vector[i][2] + p_velocity[i][2]
                p_vector[i][3] = p_vector[i][3] + p_velocity[i][3]


                if p_vector[i][0] <= x_lb:
                    p_vector[i][0] = max(x_lb, p_vector[i][0])
                elif p_vector[i][0] >= x_ub:
                    p_vector[i][0] = min(x_ub, p_vector[i][0])

                if p_vector[i][1] <= y_lb:
                    p_vector[i][1] = max(y_lb, p_vector[i][1])
                elif p_vector[i][1] >= x_ub:
                    p_vector[i][1] = min(y_ub, p_vector[i][1])

                if p_vector[i][2] <= w_lb:
                    p_vector[i][2] = max(w_lb, p_vector[i][2])
                elif p_vector[i][2] >= w_ub:
                    p_vector[i][2] = min(w_ub, p_vector[i][2])

                if p_vector[i][3] <= z_lb:
                    p_vector[i][3] = max(z_lb, p_vector[i][3])
                elif p_vector[i][3] >= z_ub:
                    p_vector[i][3] = min(z_ub, p_vector[i][3])

            best_objective_save.append(schwefel(best_particle[0], best_particle[1], best_particle[2], best_particle[3], alpha, n))
            iterations_counter += 1

        df_fitness_objectives[f'Seed {seed_it}'] = fitness_save
        df_best_objective[f'Seed {seed_it}'] = best_objective_save

    df_fitness_objectives.to_excel(f'PSO_HW6/results/fitness_total_v9_sch_{pop_it}.xlsx', index = False)
    df_best_objective.to_excel(f'PSO_HW6/results/best_objectives_v9_sch_{pop_it}.xlsx', index = False)