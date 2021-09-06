# Enconding should be a matching diagram. The movement operator is a swap in one of the two sets and keep the
# rest the same.
#%%
import numpy as np
import pandas as pd

#%%
def objective_function(distance, flow, departments_set_square, department_assignment):

    z = np.sum([flow[i, j] * distance[department_assignment[i], department_assignment[j]] for i, j in departments_set_square])

    return z

#%%
def swap_movement(first_set, second_set):

    random_swap = np.random.choice(second_set, 1, replace=False)
    #print(random_swap)
    index_one = np.where(second_set == random_swap[0])
    #print(index_one[0])
    #print(second_set)
    #second_set = np.concatenate([second_set[index_one[0][0]:],second_set[0:index_one[0][0]]])
    #print(second_set)
    index_two = index_one - np.random.choice(np.array([1, 2, 3]), 1, replace=False)
    second_set[index_one], second_set[index_two] = second_set[index_two], second_set[index_one]
    new_assignment = {first_set[i]: second_set[i] for i in first_set}

    return new_assignment

def insertion(first_set, second_set):

    random_swap = np.random.choice(second_set, 1, replace=False)
    #print(random_swap)
    index_one = np.where(second_set == random_swap[0])
    index_two = index_one - np.random.choice(np.array([1, 2]), 1, replace=False)
    second_set[index_one], second_set[index_two] = second_set[index_two], second_set[index_one]
    new_assignment = {first_set[i]: second_set[i] for i in first_set}

    return new_assignment

#%%
# Import data
df_distance = pd.read_excel('data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

n_departments = len(distance_matrix)
#%%
# Sets
arcs = np.array([(i, j) for i in range(n_departments) for j in range(n_departments)])

#%% Info Experiments

exp_seeds = [0, 3, 5, 7, 10]
exp_temp = [100, 500, 1000]
exp_stops = [1000, 5000, 10000]

sequence_final = []
save_z = []
seeds_list = []
temp_list = []
stops_list = []
#%% Experiments

for it_seed in exp_seeds:
    for it_temp in exp_temp:
        for it_stops in exp_stops:
            print("New experiment", it_seed, it_temp, it_stops)
            # Seed
            np.random.seed(it_seed)

            # Create random initial solution
            departments_set = np.array([i for i in range(n_departments)])
            location_set = np.array([i for i in range(n_departments)])
            np.random.shuffle(location_set)
            assignment_initial = {departments_set[i] : location_set[i] for i in range(n_departments)}
            assignment_stage = assignment_initial.copy()
            assignment_final = assignment_initial.copy()

            # Parameters
            number_iterations = it_stops
            temperature_0 = it_temp
            z_next = objective_function(distance_matrix, flow_matrix, arcs, assignment_initial)
            temperature_next = temperature_0
            number_moves = 25
            alpha = 0.9
            accepted_moves = 0

            temp_save = [temperature_0]
            obj_final = [z_next]

            # Simulated annealing
            for i in range(number_iterations):
                for j in range(number_moves):

                    # Do next move
                    assignment_temp = swap_movement(departments_set, location_set)

                    z_temp = objective_function(distance_matrix, flow_matrix, arcs, assignment_temp)

                    # For the evaluation if the next IF is false
                    n_random = np.random.random()
                    exp_fun = np.exp(-(z_temp - z_next)/ temperature_next)

                    if z_temp <= z_next:
                        assignment_stage = assignment_temp.copy()
                    elif n_random <= exp_fun:
                        accepted_moves += 1
                        assignment_stage = assignment_temp.copy()
                    else: # Solution remains the same
                        assignment_stage = assignment_stage

                    z_next = objective_function(distance_matrix, flow_matrix, arcs, assignment_stage)
                    z_final = objective_function(distance_matrix, flow_matrix, arcs, assignment_final)

                    if z_next <= z_final:
                        assignment_final = assignment_stage.copy()

                temperature_next = alpha * temperature_next

                temp_save.append(temperature_next)
                obj_final.append(z_final)

            print(accepted_moves)
            pd.DataFrame({'temp': temp_save, 'cost': obj_final}).to_excel(f'results_qap/exp_{it_seed}_{it_temp}_{it_stops}.xlsx')


            sequence_final.append(assignment_final.values())
            save_z.append(z_final)
            seeds_list.append(it_seed)
            temp_list.append(it_temp)
            stops_list.append(it_stops)

pd.DataFrame({'Seed': seeds_list, 'Temperature': temp_list, 'Iterations': stops_list, 'Sequence': sequence_final, 'cost': save_z}).to_excel(f'results_qap/solutions.xlsx')