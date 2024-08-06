import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 100

# Generate random data Zt from standard normal distribution
Z = np.random.normal(0, .001, n)

# Time vector
t = np.arange(1, n + 1)

# Define the time series
Zt = Z
Yt = np.cos(2 * np.pi * t / 7) + Z
Xt = np.cos(Z)
Wt = t * Z
Vt = np.cumsum(np.cos(Z))  # cumulative sum up to n

# Calculate Ut using a rolling window with exponential weights for the last 5 values
Ut = np.array([np.sum((1/2)**(np.arange(min(5, i))[::-1]) * np.cos(Z[max(0, i-5):i])) for i in range(1, n+1)])

Qt = np.cumsum((1/2)**(t-1) * np.cos(Z))  # entire series, weighted

# Create plots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
axes = axes.flatten()

# Plot each time series
axes[0].plot(t, Zt)
axes[0].set_title('Zt')

axes[1].plot(t, Yt)
axes[1].set_title('Yt = cos(2πt/7) + Zt')

axes[2].plot(t, Xt)
axes[2].set_title('Xt = cos(Zt)')

axes[3].plot(t, Wt)
axes[3].set_title('Wt = t * Zt')

axes[4].plot(t, Vt)
axes[4].set_title('Vt = Σ cos(Zi) from i=1 to t')

axes[5].plot(t, Ut)
axes[5].set_title('Ut = Σ (1/2)^(t-i) cos(Zi) from i=t-5 to t')

axes[6].plot(t, Qt)
axes[6].set_title('Qt = Σ (1/2)^(t-i) cos(Zi) from i=1 to t')

# Leave last two plots empty
axes[7].axis('off')
axes[8].axis('off')

plt.tight_layout()
plt.show()
