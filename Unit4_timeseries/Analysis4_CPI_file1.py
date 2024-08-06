import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
data = pd.read_csv('CPI.csv')
print (data.head())
# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Ensure the data is sorted by date
data.sort_values('date', inplace=True)

# Drop duplicates based on year and month, keeping only the first entry per month
data['YearMonth'] = data['date'].dt.to_period('M')
data = data.drop_duplicates(subset='YearMonth')

# Reset index after dropping duplicates
data.reset_index(drop=True, inplace=True)

# Handle missing values in the CPI column - drop or impute
data = data.dropna(subset=['CPI'])  # Option to drop rows where CPI is NaN
# Alternatively, you could fill missing values with interpolation or other methods:
# data['CPI'] = data['CPI'].interpolate()  # Example of interpolation

# Find the index of the first occurrence of July 2008
start_index = data[data['YearMonth'] == '2008-07'].index[0]

# Create a sequence of labels starting from 0
data['Time_Index'] = range(-start_index, len(data) - start_index)

# Prepare the feature (Time_Index) and the target (CPI)
X = data[['Time_Index']]  # Features need to be 2D for sklearn
y = data['CPI']

# Fit a linear model
model = LinearRegression()
model.fit(X, y)

# Calculate predictions for the fitted line
data['Predicted_CPI'] = model.predict(X)

# Calculate residuals
data['Residuals'] = data['CPI'] - data['Predicted_CPI']

# Find the maximum absolute residual
max_residual = np.max(np.abs(data['Residuals']))

# Coefficients and intercept
alpha_0 = model.intercept_
alpha_1 = model.coef_[0]

# Plotting the CPI data and the linear fit
plt.figure(figsize=(14, 7))

# Subplot for CPI and Fitted Line
plt.subplot(1, 2, 1)
plt.scatter(data['Time_Index'], data['CPI'], color='blue', label='Actual CPI')
plt.plot(data['Time_Index'], data['Predicted_CPI'], color='red', label='Fitted Line', linestyle='--')
plt.title('Linear Regression Fit to CPI')
plt.xlabel('Time Index (Months from July 2008)')
plt.ylabel('CPI')
plt.legend()
plt.grid(True)

# Subplot for Residuals
plt.subplot(1, 2, 2)
plt.stem(data['Time_Index'], data['Residuals'], linefmt='grey', markerfmt='o', basefmt=" ")
plt.title('Residuals of the Fit')
plt.xlabel('Time Index (Months from July 2008)')
plt.ylabel('Residual')
plt.grid(True)

plt.tight_layout()
plt.show()

# Output the coefficients and maximum residual
print(f"Intercept (Alpha_0): {alpha_0:.3f}")
print(f"Slope (Alpha_1): {alpha_1:.3f}")
print(f"Maximum Absolute Residual: {max_residual:.3f}")
