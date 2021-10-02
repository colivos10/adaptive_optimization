# %%
import numpy as np
import pandas as pd

# %%
def z_function(x, y):
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.exp(1)
    return z


# %%
mu = 5
lamb = 60

x_lower_bound = -32
x_upper_bound = 32
y_lower_bound = -32
y_upper_bound = 32

sigma_x = 0.05 * (x_upper_bound - x_lower_bound)
sigma_y = 0.05 * (y_upper_bound - y_lower_bound)

generations = 10
min_z = 1000
seed_exp = [i for i in range(10)]
mu_exp = [mu]
lamb_exp = [5, 10, 20]

sigma_percentage = 0.05

z_total_values = []
sigma_total = []
position_total = []

# %%

for seed_it in seed_exp:

    for mu_it in mu_exp:
        for lamb_it in lamb_exp:

            z_save = []
            sigma_save = []
            position_save = []

            sigma_x = sigma_percentage * (x_upper_bound - x_lower_bound)
            sigma_y = sigma_percentage * (y_upper_bound - y_lower_bound)

            mu = mu_it
            lamb = lamb_it

            np.random.seed(seed_it)
            x_parents = {i: np.random.uniform(low=x_lower_bound, high=x_upper_bound) for i in range(mu)}
            y_parents = {i: np.random.uniform(low=y_lower_bound, high=y_upper_bound) for i in range(mu)}

            z_parents = {i: z_function(x_parents[i], y_parents[i]) for i in range(mu)}

            children_number = int(lamb_it / mu_it)

            x_children = []
            y_children = []

            n = 3
            success_counter = 0
            best_sol = 1000
            total_counter = 0

            np.random.seed(0) # Starts randomness

            z_save.append(z_parents[0])
            sigma_save.append(sigma_x)
            position_save.append((x_parents[0], y_parents[0]))

            for gen_it in range(generations):
                x_children = []
                y_children = []
                for sol_it in range(len(x_parents)):

                    x_it = x_parents[sol_it]
                    y_it = y_parents[sol_it]

                    for iterator in range(children_number):

                        perturbation_x = np.random.normal(0, sigma_x)
                        perturbation_y = np.random.normal(0, sigma_y)
                        x_new = x_it + perturbation_x
                        y_new = x_it + perturbation_x

                        while x_new < x_lower_bound and x_new > x_upper_bound:
                            perturbation_x = np.random.normal(0, sigma_x)
                            x_new = x_it + perturbation_x
                        while y_new < y_lower_bound and y_new > y_upper_bound:
                            perturbation_y = np.random.normal(0, sigma_y)
                            y_new = y_it + perturbation_y

                        x_children.append(x_new)
                        y_children.append(y_new)

                        if z_function(x_new, y_new) < best_sol:
                            best_sol = z_function(x_new, y_new)
                            success_counter += 1

                        total_counter += 1

                        if total_counter > n:
                            if success_counter/total_counter >= 1/5 and total_counter >= 10 * n:
                                sigma_x = sigma_x/0.85
                                sigma_y = sigma_y/0.85
                                total_counter = 0
                            elif success_counter/total_counter < 1/5 and total_counter >= 10 * n:
                                sigma_x = sigma_x * 0.85
                                sigma_y = sigma_y * 0.85
                                total_counter = 0

                x_children = {i: x_children[i] for i in range(lamb)}
                y_children = {i: y_children[i] for i in range(lamb)}
                z_children = {i: z_function(x_children[i], y_children[i]) for i in range(lamb)}

                z_total = {i: z_function(x_parents[i], y_parents[i]) for i in range(mu)}
                x_total = {i: x_parents[i] for i in range(mu)}
                y_total = {i: y_parents[i] for i in range(mu)}

                for i in range(mu, mu + lamb):
                    z_total[i] = z_function(x_children[i - mu], y_children[i - mu])
                    x_total[i] = x_children[i - mu]
                    y_total[i] = y_children[i - mu]

                z_total = {k: v for k, v in sorted(z_total.items(), key=lambda item: item[1])}

                z_parents = {j: z_total[i] for i,j in list(zip(list(z_total.keys())[:mu], [k for k in range(mu)]))}
                x_parents = {j: x_total[i] for i,j in list(zip(list(z_total.keys())[:mu], [k for k in range(mu)]))}
                y_parents = {j: y_total[i] for i,j in list(zip(list(z_total.keys())[:mu], [k for k in range(mu)]))}

                z_save.append(z_parents[0])
                sigma_save.append(sigma_x)
                position_save.append((x_parents[0], y_parents[0]))

                if min(z_parents.values()) < min_z:
                    min_z = min(z_parents.values())
                    min_seed = seed_it
                    min_lamb = lamb_it
                    min_mu = mu_it
                    min_x = x_parents[0]
                    min_y = y_parents[0]

            z_total_values.append(z_save)
            sigma_total.append(sigma_save)
            position_total.append(position_save)

df_z = pd.DataFrame(z_total_values).T
df_sigma = pd.DataFrame(sigma_total).T
df_position = pd.DataFrame(position_total).T

df_z.to_excel('z_values_1.xlsx', index=False)
df_sigma.to_excel('sigma_values_1.xlsx', index=False)
df_position.to_excel('sigma_values_1.xlsx', index=False)