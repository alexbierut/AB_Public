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

# Convert and clean the data
for column in data.columns:
    data[column] = pd.to_numeric(data[column].astype(str).str.strip(), errors='coerce')
data.dropna(subset=['Year', 'Month', 'CO2'], inplace=True)

# Compute Date_year_float as the number of months since January 1958, converted to fractional years
data['Date_year_float'] = ((data['Year'] - 1958) * 12 + (data['Month'] - 1)) / 12.0

# Split the data into training and testing sets with an 80-20 split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare the training data
X_train = train_data[['Date_year_float']]
y_train = train_data['CO2']

# Prepare the test data
X_test = test_data[['Date_year_float']]
y_test = test_data['CO2']

# Instantiate and fit the linear model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
test_data['Predicted_CO2'] = model.predict(X_test)

# Compute the Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) on the test data
rmse = np.sqrt(mean_squared_error(y_test, test_data['Predicted_CO2']))
mape = mean_absolute_percentage_error(y_test, test_data['Predicted_CO2']) * 100

print("Root Mean Squared Error (RMSE) on Test Data:", rmse)
print("Mean Absolute Percentage Error (MAPE) on Test Data:", mape, "%")

# Plot actual vs predicted CO2 concentrations on test data
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual CO2')
plt.plot(X_test, test_data['Predicted_CO2'], color='red', label='Predicted CO2')
plt.title('Actual vs Predicted CO2 Concentrations on Test Data')
plt.xlabel('Year (as a floating-point number)')
plt.ylabel('CO2 Concentration (ppm)')
plt.legend()
plt.show()

# Calculate and plot residuals on test data
test_data['Residuals'] = test_data['CO2'] - test_data['Predicted_CO2']
plt.figure(figsize=(10, 6))
plt.scatter(X_test, test_data['Residuals'], color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals of CO2 Concentrations on Test Data')
plt.xlabel('Year (as a floating-point number)')
plt.ylabel('Residuals (ppm)')
plt.show()
