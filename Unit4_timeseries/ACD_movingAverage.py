import numpy as np

# Parameters for the MA(2) process
theta = [1, 0.5, 0.3333]  # Coefficients θ0, θ1, θ2
sigma_squared = 1  # Variance of white noise

# Variance calculation for gamma_X(0)
gamma_0 = sigma_squared * sum(theta_i ** 2 for theta_i in theta)
print(f"Variance gamma_X(0) = {gamma_0:.2f}")

# Autocovariance calculations using the given formula
def autocovariance(h, theta, sigma_squared):
    if h > len(theta) - 1:
        return 0
    return sigma_squared * sum(theta[j] * theta[j + h] for j in range(len(theta) - h))

gamma_1 = autocovariance(1, theta, sigma_squared)
gamma_2 = autocovariance(2, theta, sigma_squared)
gamma_3 = autocovariance(3, theta, sigma_squared)
gamma_4 = autocovariance(4, theta, sigma_squared)

print(f"gamma_X(1) = {gamma_1:.2f}")
print(f"gamma_X(2) = {gamma_2:.2f}")
print(f"gamma_X(3) = {gamma_3:.2f}")
print(f"gamma_X(4) = {gamma_4:.2f}")