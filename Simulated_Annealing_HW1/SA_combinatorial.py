# Enconding should be a matching diagram. The movement operator is a swap in one of the two sets and keep the
# rest the same.

import numpy as np
import pandas as pd

def objective_function(cost_matrix, solution_set):

    z = np.sum([cost_matrix[i,j] for (i,j) in solution_set])

    return z

def swap_movement(assignment_set):

    return assignment_set

# Import data
df_distance = pd.read_excel('Simulated_Annealing_HW1/data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('Simulated_Annealing_HW1/data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

assignment_matrix = distance_matrix * flow_matrix # Cost matrix

# Create random initial solution
departments_first_set = np.array([i for i in range(len(assignment_matrix))])
departments_second_set = np.array([i for i in range(len(assignment_matrix))])
np.random.shuffle(departments_second_set)
assignment_set = np.vstack((departments_first_set, departments_second_set)).T

# Generate initial solution randomly
#np.random.seed(0)
# Simulated annealing parameters
temperature_0 = 10
temperature_next = temperature_0

number_iterations = 10000
number_moves = 10
alpha = 0.2

x_next = x_0
y_next = y_0

x_final = x_0
y_final = y_0

z_next = z_function(x_next, y_next)

x_temp = 0
y_temp = 0

# Simulated annealing
for i in range(number_iterations):
    for j in range(number_moves):

        # Get x next solution
        x_temp = uniform_movement_operator(x_next, x_lower_bound, x_upper_bound)
        while (x_temp > x_upper_bound) or (x_temp < x_lower_bound):
            x_temp = uniform_movement_operator(x_next, x_lower_bound, x_upper_bound)

        # Get y next solution
        y_temp = uniform_movement_operator(y_next, y_lower_bound, y_upper_bound)
        while (y_temp > y_upper_bound) or (y_temp < y_lower_bound):
            y_temp = uniform_movement_operator(y_next, y_lower_bound, y_upper_bound)

        z_temp = z_function(x_temp, y_temp)

        # For the evaluation if the next IF is false
        n_random = np.random.random()
        exp_fun = np.exp(-(z_temp - z_next)/ temperature_next)

        if z_temp <= z_next:
            x_next = x_temp
            y_next = y_temp
        elif n_random <= exp_fun:
            x_next = x_temp
            y_next = y_temp
        else: # Solution remains the same
            x_next = x_next
            y_next = y_next

        z_next = z_function(x_next, y_next)
        z_final = z_function(x_final, y_final)

        if z_next <= z_final:
            x_final = x_next
            y_final = y_next

    temperature_next = alpha * temperature_0

print(x_final, y_final, z_final)