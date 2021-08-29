import numpy as np

def z_function(x, y):
    z = (4 - 2.1 * (x ** 2) + (x ** 4) / 3) * (x ** 2) + x * y + (-4 + 4 * (y ** 2)) * (y ** 2)

    return z


def uniform_movement_operator(x_null, lower_bound, upper_bound,):
    number_movement = (np.random.random() - 0.5) * (upper_bound - lower_bound)
    x_next_movement = x_null + number_movement

    return x_next_movement


def normal_movement_operator(x_null, lower_bound, upper_bound):
    number_movement = (upper_bound - lower_bound) / 6 * np.random.normal()
    x_next_movement = x_null + number_movement

    return x_next_movement

# Set variables' lower and upper bound
x_lower_bound = -3
x_upper_bound = 3
y_lower_bound = -2
y_upper_bound = 2
#np.random.seed(0)

# Generate initial solution randomly from a uniform distribution
x_0 = np.random.uniform(low=x_lower_bound, high=x_upper_bound, size=None)
y_0 = np.random.uniform(low=y_lower_bound, high=y_upper_bound, size=None)

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








