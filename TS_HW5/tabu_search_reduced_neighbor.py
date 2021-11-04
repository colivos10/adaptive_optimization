import numpy as np
import pandas as pd
import itertools
#%%
def objective_function(distance, flow, departments_set_square, department_assignment):

    z = np.sum([flow[i, j] * distance[department_assignment[i], department_assignment[j]] for i, j in departments_set_square])

    return z

#%%
def all_swap(s, distance, flow, set_square):
    swap_result = []
    swap_objective = []
    swap_indeces = []
    for idx1, idx2 in itertools.combinations(range(len(s)), 2):
        swapped_s = list(s)
        swapped_s[idx1], swapped_s[idx2] = swapped_s[idx2], swapped_s[idx1]
        swap_indeces.append((idx1, idx2))
        swap_result.append(np.array(swapped_s))
        swap_objective.append(objective_function(distance, flow, set_square, swapped_s))
    return swap_indeces, swap_result, swap_objective

#%%
# Import data
df_distance = pd.read_excel('TS_HW5/data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('TS_HW5/data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

n_departments = len(distance_matrix)
#%%
# Sets
arcs = np.array([(i, j) for i in range(n_departments) for j in range(n_departments)])

# Parameters
number_iterations = 50
tabu_length = 13
obj_final_total = []

#%%
for tabu_exp in [13]:
    tabu_length = tabu_exp
    for exp_it in range(10):
        np.random.seed(exp_it)
        departments_set = np.array([i for i in range(n_departments)])
        location_set = np.array([i for i in range(n_departments)])
        np.random.shuffle(location_set)
        assignment_initial = {departments_set[i] : location_set[i] for i in range(n_departments)}
        assignment_stage = assignment_initial.copy()
        assignment_final = assignment_initial.copy()
        best_obj = objective_function(distance_matrix, flow_matrix, arcs, assignment_initial)
        tabu_list = []
        obj_final = [best_obj]
        best_set = location_set
        for i in range(number_iterations):
            result_swap, solutions_swap, objective_swap = all_swap(location_set, distance_matrix, flow_matrix, arcs)
            neighborhood = np.array([[result_swap[j], solutions_swap[j], objective_swap[j]] for j in range(190)])
            neighborhood = neighborhood[neighborhood[:, 2].argsort()]

            index_neighborhood = np.random.randint(0, 189)

            neigh_sol = neighborhood[index_neighborhood]

            #if neigh_sol[0] not in tabu_list:
            #    break
            #elif neigh_sol[2] < best_obj:
            #    best_obj = neigh_sol[2]
            #    best_set = neigh_sol[1]
            #    break

            location_set = neigh_sol[1]

            tabu_list.insert(0, neigh_sol[0])
            if len(tabu_list) > tabu_length:
                tabu_list.pop(-1)

            if neigh_sol[2] < best_obj:
                best_obj = neigh_sol[2]
                best_set = neigh_sol[1]

            obj_final.append(best_obj)
        obj_final_total.append(obj_final)
        print(best_set)

df = pd.DataFrame(obj_final_total)
df.T.to_excel("TS_HW5/obj_value_e.xlsx")