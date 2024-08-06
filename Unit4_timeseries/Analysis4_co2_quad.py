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

#Drop -99.99 from co2 and convert to numbers, drop nans
data['CO2'] = data['CO2'].astype(str).str.strip()
data = data[data['CO2'] != '-99.99']
for column in data.columns:
    data[column] = pd.to_numeric(data[column].astype(str).str.strip(), errors='coerce')
data.dropna(subset=['Year', 'Month', 'CO2'], inplace=True)
data ["time"] = data["Date_year_float"]-1958
# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)

# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(train_data[['time']])
y_train = train_data['CO2']
X_test = poly.transform(test_data[['time']])
y_test = test_data['CO2']

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
train_data['Predicted_CO2'] = model.predict(X_train)
test_data['Predicted_CO2'] = model.predict(X_test)

# Performance metrics
rmse = np.sqrt(mean_squared_error(y_test, test_data['Predicted_CO2']))
mape = mean_absolute_percentage_error(y_test, test_data['Predicted_CO2']) * 100

# Output results
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape, "%")
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Plot actual vs predicted CO2 concentrations
plt.figure(figsize=(10, 6))
plt.scatter(train_data['time'], y_train, color='blue', label='Actual CO2')
plt.plot(train_data['time'], train_data['Predicted_CO2'], color='red', label='Predicted CO2')
plt.title('Actual vs Predicted CO2 Concentrations')
plt.xlabel('Year (as a floating-point number)')
plt.ylabel('CO2 Concentration (ppm)')
plt.legend()
plt.show()

# Residuals plot
train_data['Residuals'] = y_train - train_data['Predicted_CO2']
plt.figure(figsize=(10, 6))
plt.scatter(train_data['time'], train_data['Residuals'], color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals of CO2 Concentrations')
plt.xlabel('Year (as a floating-point number)')
plt.ylabel('Residuals (ppm)')
plt.show()
