import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('CPI.csv')

# Convert 'date' column to datetime and sort by date
data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True)

# Drop duplicates based on year and month, keeping only the first entry per month
data['YearMonth'] = data['date'].dt.to_period('M')
data = data.drop_duplicates(subset='YearMonth')

# Reset index after dropping duplicates
data.reset_index(drop=True, inplace=True)

# Handle missing values in the CPI column - drop or impute
data = data.dropna(subset=['CPI'])  # Option to drop rows where CPI is NaN

# Find the index of the first occurrence of July 2008
start_index = data[data['YearMonth'] == '2008-07'].index[0]

# Create a sequence of labels starting from 0
data['Time_Index'] = range(-start_index, len(data) - start_index)

# Filter the data to train up to September 2013
training_data = data[data['date'] < '2013-09'].copy()
test_data = data[data['date'] >= '2013-09'].copy()
# Fit a linear regression to get residuals
X_train = training_data[['Time_Index']]
y_train = training_data['CPI']
X_test = test_data[['Time_Index']]
y_test = test_data['CPI']
model = LinearRegression()
model.fit(X_train, y_train)

data['Predicted_CPI'] = model.predict(data[['Time_Index']])
# Residuals for the entire dataset
data['Residuals'] = data['CPI'] - (model.intercept_ + model.coef_[0] * data['Time_Index'])

# Avoid SettingWithCopyWarning by using .loc for safe assignment
training_data.loc[:, 'Predicted_CPI'] = model.predict(X_train)
training_data.loc[:, 'Residuals'] = training_data['CPI'] - training_data['Predicted_CPI']

test_data.loc[:, 'Predicted_CPI'] = model.predict(X_test)
test_data.loc[:, 'Residuals'] = test_data['CPI'] - test_data['Predicted_CPI']

# Initialize lists to store forecasts and actual values
# Define the rebuild_diffed function
# Define the rebuild_diffed function
def rebuild_diffed(series, linear_trend):
    return series + linear_trend

lag = 2  # Assuming a lag of 2 based on previous analysis
ar_model = AutoReg(training_data['Residuals'], lags=lag, trend='n')
ar_model_fit = ar_model.fit()

# Predict the residuals dynamically
test_residuals_predicted = ar_model_fit.predict(start=len(training_data), end=len(training_data) + len(test_data) - 1, dynamic=True)

# Use the linear trend to rebuild the CPI values
test_data['LinearTrend'] = model.predict(X_test)
test_data['CPI_Reconstructed'] = rebuild_diffed(test_residuals_predicted.values, test_data['LinearTrend'])

# Calculate the RMSE for the test period
rmse = np.sqrt(mean_squared_error(test_data['CPI'], test_data['CPI_Reconstructed']))
print(f'RMSE for the test period: {rmse:.3f}')

## Plot the actual vs. forecasted CPI for the test period
#plt.figure(figsize=(10, 5))
#plt.plot(data['date'][len(training_data):], actuals, label='Actual CPI', color='blue')
#plt.plot(data['date'][len(training_data):], forecasts, label='Forecasted CPI', color='red', linestyle='--')
#plt.title('Actual vs. Forecasted CPI')
#plt.xlabel('Date')
#plt.ylabel('CPI')
#plt.legend()
#plt.grid(True)
#plt.show()
