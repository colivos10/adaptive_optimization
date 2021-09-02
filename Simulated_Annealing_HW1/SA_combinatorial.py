# Enconding should be a matching diagram. The movement operator is a swap in one of the two sets and keep the
# rest the same.

import numpy as np
import pandas as pd

def objective_function(dist_matrix, flow_matrix, departments_set, solution_set):

    z = np.sum([(np.sum([dist_matrix[j, k] * flow_matrix[i, k] for k in departments_set])) for (i,j) in solution_set])

    return z

def swap_movement(first_set, second_set):

    random_swap = np.random.choice(second_set, 2)

    index_one = np.where(second_set == random_swap[0])
    index_two =np.where(second_set == random_swap[1])
    second_set[index_one], second_set[index_two] = second_set[index_two], second_set[index_one]
    assignment_set = np.vstack((first_set, second_set)).T

    return assignment_set

# Import data
df_distance = pd.read_excel('data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

# Create random initial solution
departments_first_set = np.array([i for i in range(len(distance_matrix))])
departments_second_set = np.array([i for i in range(len(distance_matrix))])
np.random.shuffle(departments_second_set)

# Solution
assignment_set_initial = np.vstack((departments_first_set, departments_second_set)).T
assignment_set_next = np.copy(assignment_set_initial)
assignment_set_temp = np.copy(assignment_set_initial)
assignment_set_final = np.copy(assignment_set_initial)

z_next = objective_function(distance_matrix, flow_matrix, departments_first_set, assignment_set_initial)
print(z_next)
# Generate initial solution randomly
np.random.seed(0)
# Simulated annealing parameters
temperature_0 = 10000
temperature_next = temperature_0

number_iterations = 1000
number_moves = 100
alpha = 0.9

z_next = objective_function(distance_matrix, flow_matrix, departments_first_set, assignment_set_next)
print(z_next)
# Simulated annealing
for i in range(number_iterations):
    for j in range(number_moves):
        #print('First', assignment_set_next)
        # Do next move
        assignment_set_temp = swap_movement(departments_first_set, departments_second_set)
        #print('Second', assignment_set_temp)
        z_temp = objective_function(distance_matrix, flow_matrix, departments_first_set, assignment_set_temp)

        # For the evaluation if the next IF is false
        n_random = np.random.random()
        exp_fun = np.exp(-(z_temp - z_next)/ temperature_next)

        if z_temp <= z_next:
            assignment_set_next = np.copy(assignment_set_temp)
        elif n_random <= exp_fun:
            assignment_set_next = np.copy(assignment_set_temp)
        else: # Solution remains the same
            assignment_set_next = assignment_set_next

        z_next = objective_function(distance_matrix, flow_matrix, departments_first_set, assignment_set_next)
        z_final = objective_function(distance_matrix, flow_matrix, departments_first_set, assignment_set_final)

        if z_next <= z_final:
            assignment_set_final = np.copy(assignment_set_next)

    temperature_next = alpha * temperature_0
    #print(z_final)
print(assignment_set_final)
print(z_final)