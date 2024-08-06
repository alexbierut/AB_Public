import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Load the data
skip_rows = 56
col_titles = [
    'Year', 'Month', 'Date', 'Date_year_float', 'CO2', 'Seasonally_Adjusted',
    'Fit', 'Seasonally_Adjusted_Fit', 'CO2_Filled', 'Seasonally_Adjusted_Filled'
]
data = pd.read_csv('CO2.csv', skiprows=skip_rows, names=col_titles)

# Drop -99.99 from CO2 and convert to numbers, drop NaNs
data['CO2'] = data['CO2'].astype(str).str.strip()
data = data[data['CO2'] != '-99.99']
for column in data.columns:
    data[column] = pd.to_numeric(data[column].astype(str).str.strip(), errors='coerce')
data.dropna(subset=['Year', 'Month', 'CO2'], inplace=True)
data["time"] = data["Date_year_float"] - 1958

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)

# Polynomial Features
poly = PolynomialFeatures(degree=2)

# Train set transformation and fitting
X_train = poly.fit_transform(train_data[['time']])
y_train = train_data['CO2']
model_train = LinearRegression()
model_train.fit(X_train, y_train)
train_data['Predicted_CO2'] = model_train.predict(X_train)
train_data['Residuals'] = train_data['CO2'] - train_data['Predicted_CO2']
train_monthly_averages = train_data.groupby('Month')['Residuals'].mean()

# Test set transformation and prediction
X_test = poly.transform(test_data[['time']])
y_test = test_data['CO2']
test_data['Predicted_CO2'] = model_train.predict(X_test)
test_data['Residuals'] = test_data['CO2'] - test_data['Predicted_CO2']
test_monthly_averages = test_data.groupby('Month')['Residuals'].mean()

# Output results
print("Training Data - Periodic Signal for January (P_Jan): {:.5f}".format(train_monthly_averages.loc[1]))
print("Training Data - Periodic Signal for February (P_Feb): {:.5f}".format(train_monthly_averages.loc[2]))
print("Testing Data - Periodic Signal for January (P_Jan): {:.5f}".format(test_monthly_averages.loc[1]))
print("Testing Data - Periodic Signal for February (P_Feb): {:.5f}".format(test_monthly_averages.loc[2]))

# Optionally: Plot the results for visual verification
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(train_monthly_averages.index, train_monthly_averages, color='skyblue')
plt.title('Average Residuals by Month (Training Data)')
plt.xlabel('Month')
plt.ylabel('Average Residuals (ppm)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(test_monthly_averages.index, test_monthly_averages, color='lightgreen')
plt.title('Average Residuals by Month (Testing Data)')
plt.xlabel('Month')
plt.ylabel('Average Residuals (ppm)')
plt.grid(True)

plt.tight_layout()
plt.show()
