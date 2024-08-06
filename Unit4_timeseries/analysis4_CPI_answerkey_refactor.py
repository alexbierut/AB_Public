import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load and preprocess data
ps_cpi = pd.read_csv('CPI.csv')
ps_cpi['date'] = pd.to_datetime(ps_cpi['date'])
ps_cpi['YearMonth'] = ps_cpi['date'].dt.strftime('%Y-%m')
cpi = ps_cpi.drop_duplicates('YearMonth', keep='last').reset_index(drop=True)

# Drop rows with NaN values
cpi.dropna(subset=['CPI'], inplace=True)

# Split data into training and test sets
cpi_train = cpi[cpi.YearMonth < '2013-09'].copy()
cpi_test = cpi[cpi.YearMonth >= '2013-09'].copy()

# Define and fit the ARIMA model (p, d, q) parameters
p, d, q = 2, 0, 0  # AR(2) model with linear trend
model = ARIMA(cpi_train.CPI, order=(p, d, q), trend='t')
arima_result = model.fit()

# Make predictions
train_predictions = arima_result.fittedvalues
test_predictions = arima_result.predict(start=len(cpi_train), end=len(cpi_train) + len(cpi_test) - 1)

# Evaluate the model
rmse = mean_squared_error(cpi_test.CPI, test_predictions) ** 0.5
print(f"The RMSE of the final fit is {rmse}")
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(cpi_train.index, cpi_train.CPI, label='Training Data')
plt.plot(cpi_test.index, cpi_test.CPI, label='Test Data')
plt.plot(cpi_train.index, train_predictions, label='Training Predictions', color='red')
plt.plot(cpi_test.index, test_predictions, label='Test Predictions', color='green')
plt.xlabel('Time')
plt.ylabel('CPI')
plt.legend()
plt.show()
