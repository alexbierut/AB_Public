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

# Fit a linear regression to get residuals
X_train = training_data[['Time_Index']]
y_train = training_data['CPI']
model = LinearRegression()
model.fit(X_train, y_train)

data['Predicted_CPI'] = model.predict(data[['Time_Index']])
# Residuals for the entire dataset
data['Residuals'] = data['CPI'] - (model.intercept_ + model.coef_[0] * data['Time_Index'])

# Avoid SettingWithCopyWarning by using .loc for safe assignment
training_data.loc[:, 'Predicted_CPI'] = model.predict(X_train)
training_data.loc[:, 'Residuals'] = training_data['CPI'] - training_data['Predicted_CPI']

# Initialize lists to store forecasts and actual values
forecasts = []
actuals = []

# Perform rolling forecasts
for i in range(len(training_data), len(data)):
    # Refit the AR model with data up to the current point
    model_ar = AutoReg(data['Residuals'][:i], lags=2, trend='c')
    model_ar_fit = model_ar.fit()

    # Predict the next step
    forecast = model_ar_fit.predict(start=i, end=i).iloc[0]

    # Store the forecast and actual value
    forecasts.append(forecast + data['Predicted_CPI'].iloc[i])
    actuals.append(data['CPI'].iloc[i])

# Calculate the RMSE for 1-month-ahead forecasts
rmse = np.sqrt(mean_squared_error(actuals, forecasts))
print(f'RMSE for 1-month-ahead forecasts: {rmse:.3f}')

# Plot the actual vs. forecasted CPI for the test period
plt.figure(figsize=(10, 5))
plt.plot(data['date'][len(training_data):], actuals, label='Actual CPI', color='blue')
plt.plot(data['date'][len(training_data):], forecasts, label='Forecasted CPI', color='red', linestyle='--')
plt.title('Actual vs. Forecasted CPI')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()
plt.grid(True)
plt.show()
