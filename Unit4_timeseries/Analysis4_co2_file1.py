import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Define the number of rows to skip based on non-data header rows in the CSV
skip_rows = 56
# Define column titles for clarity and proper data handling
col_titles = [
    'Year', 'Month', 'Date', 'Date_year_float', 'CO2', 'Seasonally_Adjusted',
    'Fit', 'Seasonally_Adjusted_Fit', 'CO2_Filled', 'Seasonally_Adjusted_Filled'
]
# Load the CSV file, skipping initial non-data rows and using specified columns
data = pd.read_csv('CO2.csv', skiprows=skip_rows, names=col_titles)

#Drop -99.99 from co2 and convert to numbers, drop nans
data['CO2'] = data['CO2'].astype(str).str.strip()
data = data[data['CO2'] != '-99.99']
for column in data.columns:
    data[column] = pd.to_numeric(data[column].astype(str).str.strip(), errors='coerce')
data.dropna(subset=['Year', 'Month', 'CO2'], inplace=True)
data ["time"] = data["Date_year_float"]-1958
print(data.head)

# Split the data into training and testing sets with an 80-20 split
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False,random_state=42)

# Prepare the data by reshaping X into a 2D array using DataFrame
X_train = train_data[['time']]  # Predictor as a DataFrame to keep it 2D
y_train = train_data['CO2']                # Response
X_test= test_data[['time']]  # Predictor as a DataFrame to keep it 2D
y_test = test_data['CO2']
# Instantiate and fit the linear model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training data
train_data['Predicted_CO2'] = model.predict(X_train)
test_data['Predicted_CO2'] = model.predict(X_test)
print (test_data)
print (train_data)
# Compute the Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE)
rmse = np.sqrt(mean_squared_error(y_test, test_data['Predicted_CO2']))
mape = mean_absolute_percentage_error(y_test, test_data['Predicted_CO2']) * 100

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape, "%")
print("Model Coefficient (Slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)

# Plot the actual vs predicted CO2 concentrations
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Actual CO2')
plt.plot(X_train, train_data['Predicted_CO2'], color='red', label='Predicted CO2')
plt.title('Actual vs Predicted CO2 Concentrations')
plt.xlabel('Year (as a floating-point number)')
plt.ylabel('CO2 Concentration (ppm)')
plt.legend()
plt.show()

# Calculate and plot residuals
train_data['Residuals'] = train_data['CO2'] - train_data['Predicted_CO2']
plt.figure(figsize=(10, 6))
plt.scatter(X_train, train_data['Residuals'], color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals of CO2 Concentrations')
plt.xlabel('Year (as a floating-point number)')
plt.ylabel('Residuals (ppm)')
plt.show()

