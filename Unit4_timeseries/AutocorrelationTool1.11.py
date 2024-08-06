import numpy as np


def autocovariance_function(time_series, lag):
    n = len(time_series)
    mean = np.mean(time_series)
    covariance = 0

    for t in range(n - lag):
        covariance += (time_series[t] - mean) * (time_series[t + lag] - mean)

    return covariance / n


# Example time series
time_series = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
time_series2 = [-1,0,1,0,-1,0,1,0]

# Calculate and print autocovariance for different lags
for h in range(6):
    gamma_h = autocovariance_function(time_series2, h)
    print(f"γ({h}) = {gamma_h:.4f}")