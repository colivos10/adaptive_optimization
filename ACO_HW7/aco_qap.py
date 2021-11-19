import numpy as np
import pandas as pd

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
# Create initial tau matrix

initial_tau = 1/(20 * 6000)
tau_matrix = np.zeros((20, 20))
for i in range(n_departments):
    for j in range(n_departments):
        if i == j:
            tau_matrix[i, j] = 0
        else:
            tau_matrix[i, j] = initial_tau

m = 10 # Number of ants
alpha = 6
beta = 6
rho = 0.85
max_iterations = 500

ants = np.empty((m,),dtype=object)

# Construct ants
def built_ants():
    for ant_counter in range(m):

        dpt_perm = np.array([i for i in range(20)]) # It contains all the departments

        first_dpt = np.random.choice(dpt_perm) # Select randomly a departments
        ant_perm = np.zeros(shape=(20,), dtype='int32')
        ant_perm[0] = first_dpt
        dpt_perm = np.delete(dpt_perm, [first_dpt])

        pbb_trail_size = 19
        pbb_trail = np.zeros(shape=(pbb_trail_size))
        counter_ant_trail = int(1)
        while sum(ant_perm) != sum(i for i in range(n_departments)):

            count = 0

            for k in dpt_perm:
                pbb_trail[count] = (tau_matrix[int(ant_perm[counter_ant_trail-1]), k] ** alpha) * (1/distance_matrix[int(ant_perm[counter_ant_trail-1]), k]) ** beta / \
                                   sum((tau_matrix[int(ant_perm[counter_ant_trail-1]), j] ** alpha) * (1 / distance_matrix[int(ant_perm[counter_ant_trail-1]), j]) ** beta for j in dpt_perm)
                count = count + 1

            cum_pbb_trail = np.cumsum(pbb_trail)

            random_value = np.random.random()
            for value_range in cum_pbb_trail:
                if random_value < value_range:
                    index_located = np.where(cum_pbb_trail == value_range)
                    index_located = index_located[0][0]
                    break

            ant_perm[counter_ant_trail] = dpt_perm[index_located]

            dpt_perm = np.delete(dpt_perm, np.where(dpt_perm == dpt_perm[index_located]))

            counter_ant_trail = counter_ant_trail + 1
            pbb_trail_size = pbb_trail_size - 1
            pbb_trail = np.zeros(shape=(pbb_trail_size))

        ants[ant_counter] = ant_perm

    return ants

# %%
# Here the algorithm starts
best_value_total = []
for seed_it in range(10):
    np.random.seed(seed_it)
    ants = built_ants()

    best_ant = 0
    best_value = 1e10
    departments_set = np.array([i for i in range(n_departments)])
    best_value_list = []

    for iterations in range(max_iterations):

        obj_value = {}
        for ant_it in range(m):
            current_value = objective_function(distance_matrix, flow_matrix, arcs, [(departments_set[pos], ants[ant_it][pos]) for pos in range(n_departments)])
            obj_value[ant_it] = current_value
            if current_value < best_value:
                best_value = current_value
                best_ant = ant_it
        best_value_list.append(best_value)
        delta_tau_matrix = np.zeros(shape=(20, 20))

        for i in range(n_departments):
            for j in range(n_departments):
                if i == j:
                    delta_tau_matrix[i, j] = 0
                else:
                    for m_it in range(m):
                        if set([i, j]).issubset(set(ants[m_it])):
                            delta_tau_matrix[i, j] = delta_tau_matrix[i, j] + (1 / obj_value[m_it])

        for i in range(n_departments):
            for j in range(n_departments):
                if i == j:
                    tau_matrix[i, j] = 0
                else:
                    tau_matrix[i, j] = rho * tau_matrix[i, j] + (1 - rho) * delta_tau_matrix[i, j]

        ants = built_ants()

    best_value_total.append(best_value_list)

df = pd.DataFrame(best_value_total)
df.T.to_excel("ACO_HW7/obj_value_a.xlsx")