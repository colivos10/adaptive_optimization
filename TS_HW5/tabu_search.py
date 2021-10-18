import numpy as np
import pandas as pd

#%%
def objective_function(distance, flow, departments_set_square, department_assignment):

    z = np.sum([flow[i, j] * distance[department_assignment[i], department_assignment[j]] for i, j in departments_set_square])

    return z

#%%
def swap_movement(first_set, second_set):

    random_swap = np.random.choice(second_set, 1, replace=False)
    index_one = np.where(second_set == random_swap[0])
    index_two = index_one - np.random.choice(np.array([1, 2, 3]), 1, replace=False)
    second_set[index_one], second_set[index_two] = second_set[index_two], second_set[index_one]
    new_assignment = {first_set[i]: second_set[i] for i in first_set}

    return new_assignment

def insertion(first_set, second_set):

    random_swap = np.random.choice(second_set, 1, replace=False)
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