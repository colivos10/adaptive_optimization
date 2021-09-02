import numpy as np
import pandas as pd
from gurobipy import *

df_distance = pd.read_excel('Simulated_Annealing_HW1/data_qap.xlsx', sheet_name = 'Distance', header=None)
distance_matrix = df_distance.to_numpy()

df_flow = pd.read_excel('Simulated_Annealing_HW1/data_qap.xlsx', sheet_name = 'Flow', header=None)
flow_matrix = df_flow.to_numpy()

N = [i for i in range(len(distance_matrix))]
L = [i for i in range(len(distance_matrix))]
A = [(i, j) for i in range(len(distance_matrix)) for j in range(len(distance_matrix))]

mdl = Model()
x = mdl.addVars(A, vtype=GRB.BINARY, name='x')
#mdl.setObjective(quicksum(x[i, j] * quicksum(distance_matrix[j, k] * flow_matrix[i, k] for k in N) for (i,j) in A), GRB.MINIMIZE)

mdl.setObjective(quicksum(x[i, k] * x[j, p] * distance_matrix[k, p] * flow_matrix[i, j] for i in N for j in N for k in N for p in N), GRB.MINIMIZE)
mdl.addConstrs(quicksum(x[i, j] for i in N) == 1 for j in L)
mdl.addConstrs(quicksum(x[i, j] for j in L) == 1 for i in N)

x[0, 3].start = 1
x[1, 10].start = 1
x[2, 4].start = 1
x[3,  0].start = 1
x[4,  1].start = 1
x[5,  9].start = 1
x[6,  8].start = 1
x[7,  5].start = 1
x[8,  11].start = 1
x[ 9, 14].start = 1
x[10,  7].start = 1
x[11, 13].start = 1
x[12,  6].start = 1
x[13, 12].start = 1
x[14,  2].start = 1

mdl.Params.TimeLimit = 300
mdl.optimize()

sol = [(i, j) for i,j in A if x[i, j].x >= 0.9]
print(sol)