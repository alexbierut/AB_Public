import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the CPI data
data = pd.read_csv('CPI.csv')
data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True)
data['YearMonth'] = data['date'].dt.to_period('M')
data.drop_duplicates(subset='YearMonth', inplace=True)
data.reset_index(drop=True, inplace=True)

# Determine index for July 2008 as the base for Time_Index
start_index = data[data['YearMonth'] == '2008-07'].index[0]
data['Time_Index'] = range(-start_index, len(data) - start_index)

# Select training data up to (not including) September 2013
training_data = data[data['date'] < '2013-09'].copy()

# Linear regression to detrend the data
X_train = training_data[['Time_Index']].values
y_train = training_data['CPI'].values
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
training_data['Predicted_CPI'] = lin_reg.predict(X_train)
training_data['Residuals'] = training_data['CPI'] - training_data['Predicted_CPI']

# Plotting the CPI and the fitted line
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(training_data['date'], training_data['CPI'], label='Actual CPI')
plt.plot(training_data['date'], training_data['Predicted_CPI'], label='Fitted Line', linestyle='--')
plt.title('Detrended CPI')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()

# Plotting the residuals
plt.subplot(122)
plt.plot(training_data['date'], training_data['Residuals'], label='Residuals')
plt.title('Residuals of CPI Fit')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.tight_layout()
plt.show()

# ACF and PACF plots to determine the order of the AR model
plot_acf(training_data['Residuals'], lags=20, title='ACF for Residuals')
plot_pacf(training_data['Residuals'], lags=20, title='PACF for Residuals')
plt.show()

# Determine the optimal number of lags using AIC
results = []
for p in range(1, 21):  # Check up to 20 lags
    model = AutoReg(training_data['Residuals'], lags=p)
    model_fit = model.fit()
    results.append({
        'lag': p,
        'aic': model_fit.aic,
        'bic': model_fit.bic,
        'hqic': model_fit.hqic
    })

results_df = pd.DataFrame(results)
best_model_info = results_df.loc[results_df['aic'].idxmin()]
best_lag = best_model_info['lag']

# Fit the best AR model
best_ar_model = AutoReg(training_data['Residuals'], lags=best_lag)
best_ar_model_fit = best_ar_model.fit()

# Print the best model results and summary
print(f"Best AR Model Lag: {best_lag}")
print(best_ar_model_fit.summary())

# Visualize AR model predictions
predictions = best_ar_model_fit.predict(start=best_lag, end=len(training_data) - 1, dynamic=False)
plt.figure(figsize=(10, 5))
plt.plot(training_data['Time_Index'][best_lag:], training_data['Residuals'][best_lag:], label='Actual Residuals')
plt.plot(training_data['Time_Index'][best_lag:], predictions, label='Predicted by AR Model', linestyle='--')
plt.title('AR Model Predictions')
plt.xlabel('Time Index (from July 2008)')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()
