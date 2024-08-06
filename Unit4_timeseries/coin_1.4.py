import numpy as np

# Initial conditions
X_0 = 0
coin_flips = ['H', 'T', 'T', 'T', 'H', 'H', 'T', 'T', 'T']  # Corrected first 9 coin flips

# Convert coin flips to +1 for H and -1 for T
steps = [1 if flip == 'H' else -1 for flip in coin_flips]

# Compute the first 10 terms of the time series
X = [X_0]
for step in steps:
    X.append(X[-1] + step)

# Print the first 10 terms of the time series
print("First 10 terms of the time series: ", X)

# Expected position at t = 10
E_X10 = X_0
print("Expected position E[X10] at time t = 10: ", E_X10)

# Expected position at t = 20
# For a random walk without drift, the expected position remains the same
E_X20 = E_X10
print("Expected position E[X20] at time t = 20: ", E_X20)

# Variance of the position at t = 10
# For a random walk, variance increases linearly with time
# Variance at t is t * variance of steps
variance_step = 1  # For fair coin flips
Var_X10 = 10 * variance_step
print("Variance of the position X10 at time t = 10: ", Var_X10)

# Variance of the position at t = 20
Var_X20 = 20 * variance_step
print("Variance of the position X20 at time t = 20: ", Var_X20)

# Forecast E[X10 | X9]
X9 = X[-1]  # Last value in the series
forecast_X10_given_X9 = X9
print("Forecast E[X10 | X9]: ", forecast_X10_given_X9)
