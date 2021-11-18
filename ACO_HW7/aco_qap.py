import numpy as np
import pandas as pd
import itertools
#%%
def objective_function(distance, flow, departments_set_square, department_assignment):

    z = np.sum([flow[i, j] * distance[department_assignment[i], department_assignment[j]] for i, j in departments_set_square])

    return z

#%%
# Import data
df_distance = pd.read_excel('TS_HW5/data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('TS_HW5/data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

n_departments = len(distance_matrix)
arcs = np.array([(i, j) for i in range(n_departments) for j in range(n_departments)])
#%%
dpt_perm = np.array([i for i in range(20)])

first_dpt = np.random.choice(dpt_perm)
ant_perm = np.array([first_dpt])
dpt_perm = np.delete(dpt_perm, [first_dpt])

initial_tau = 1/(20 * 2000)

# Create initial tau matrix
tau_matrix = np.zeros((20, 20))
for i in range(n_departments):
    for j in range(n_departments):
        if i == j:
            tau_matrix[i, j] = 0
        else:
            tau_matrix[i, j] = initial_tau

alpha = 1
beta = 2
k = 2
for i in dpt_perm:
    probability = (tau_matrix[0, k] ** alpha) * (1/distance_matrix[0, k]) ** beta / sum((tau_matrix[i, j] ** alpha) * (1/distance_matrix[i, j]) ** beta for (i, j) in arcs if i != j)




