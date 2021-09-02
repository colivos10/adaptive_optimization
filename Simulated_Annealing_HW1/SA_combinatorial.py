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

    random_swap = np.random.choice(second_set, 2, replace=False)
    #print(random_swap)
    index_one = np.where(second_set == random_swap[0])
    index_two =np.where(second_set == random_swap[1])
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

#%%
# Create random initial solution
np.random.seed(1)
departments_set = np.array([i for i in range(n_departments)])
location_set = np.array([i for i in range(n_departments)])
#location_set = np.array([4, 3, 7, 2, 6, 0, 12, 13, 9, 10, 14, 11, 8, 1, 5])
np.random.shuffle(location_set)

#%%
# Solution
assignment_initial = {departments_set[i] : location_set[i] for i in range(n_departments)}
assignment_stage = assignment_initial.copy()
assignment_final = assignment_initial.copy()

#%%
# Objective function
z_next = objective_function(distance_matrix, flow_matrix, arcs, assignment_initial)
print(z_next)
# Generate initial solution randomly
#np.random.seed(0)
# Simulated annealing parameters
temperature_0 = 250
temperature_next = temperature_0

number_iterations = 10000
number_moves = 50
alpha = 0.9

accepted_moves = 0

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

    temperature_next = alpha * temperature_0
    #print(z_final)
print(assignment_final)
print(z_final)
print(accepted_moves)