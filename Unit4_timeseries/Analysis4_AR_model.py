import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
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

# Avoid SettingWithCopyWarning by using .loc for safe assignment
training_data.loc[:, 'Predicted_CPI'] = model.predict(X_train)
training_data.loc[:, 'Residuals'] = training_data['CPI'] - training_data['Predicted_CPI']

# Plot ACF and PACF to determine the order p of the AR model
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(training_data['Residuals'], lags=20, ax=plt.gca(), title='ACF for Residuals')
plt.subplot(122)
plot_pacf(training_data['Residuals'], lags=20, ax=plt.gca(), title='PACF for Residuals')
plt.tight_layout()
plt.show()

# Assuming AR(1) based on PACF
model_ar = AutoReg(training_data['Residuals'], lags=2)
model_ar_fit = model_ar.fit()

# Print the summary of the AR model
print(model_ar_fit.summary())

# Avoid SettingWithCopyWarning by using .loc for safe assignment
training_data.loc[:, 'AR_Predictions'] = model_ar_fit.predict(start=1, end=len(training_data))
training_data.loc[:, 'AR_Residuals'] = training_data['Residuals'] - training_data['AR_Predictions']

# Plotting the AR model predictions
plt.figure(figsize=(10, 5))
plt.plot(training_data['Time_Index'], training_data['Residuals'], label='Original Residuals')
plt.plot(training_data['Time_Index'], training_data['AR_Predictions'], label='AR Predictions', color='red')
plt.title('AR Model Predictions vs. Original Residuals')
plt.xlabel('Time Index')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residuals of the AR model
plt.figure(figsize=(10, 5))
plt.stem(training_data['Time_Index'], training_data['AR_Residuals'], linefmt='grey', markerfmt='o', basefmt=" ")
plt.title('Residuals of the AR Model')
plt.xlabel('Time Index')
plt.ylabel('Residual Value')
plt.grid(True)
plt.show()
