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

# Ensure there are no missing values in the CPI column
data.dropna(subset=['CPI'], inplace=True)

# Determine index for July 2008 as the base for Time_Index
start_index = data[data['YearMonth'] == '2008-07'].index[0]
data['Time_Index'] = range(-start_index, len(data) - start_index)

# Fit a linear regression to the entire dataset
X = data[['Time_Index']].values
y = data['CPI'].values
lin_reg = LinearRegression()
lin_reg.fit(X, y)
data['Predicted_CPI'] = lin_reg.predict(X)
data['Residuals'] = data['CPI'] - data['Predicted_CPI']

# Plotting the CPI and the fitted line
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['CPI'], label='Actual CPI')
plt.plot(data['date'], data['Predicted_CPI'], label='Fitted Line', linestyle='--')
plt.title('Detrended CPI Across Entire Dataset')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()
plt.show()

# Plotting the residuals
plt.plot(data['date'], data['Residuals'], label='Residuals')
plt.title('Residuals of CPI Fit')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.show()

# ACF and PACF plots to determine the order of the AR model
plot_acf(data['Residuals'], lags=20, title='ACF for Residuals')
plot_pacf(data['Residuals'], lags=20, title='PACF for Residuals')
plt.show()

# Determine the optimal number of lags using AIC
results = []
for p in range(1, 21):  # Check up to 20 lags
    model = AutoReg(data['Residuals'], lags=p, trend='n')
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

# Fit the best AR model with no trend
best_ar_model = AutoReg(data['Residuals'], lags=best_lag, trend='n')
best_ar_model_fit = best_ar_model.fit()

# Print the best model parameters directly
print(f"Best AR Model Lag: {best_lag}")
print("Model Coefficients:", best_ar_model_fit.params)

# Visualize AR model predictions
predictions = best_ar_model_fit.predict(start=best_lag, end=len(data) - 1, dynamic=False)
plt.figure(figsize=(10, 5))
plt.plot(data['Time_Index'][best_lag:], data['Residuals'][best_lag:], label='Actual Residuals')
plt.plot(data['Time_Index'][best_lag:], predictions, label='Predicted by AR Model', linestyle='--')
plt.title('AR Model Predictions')
plt.xlabel('Time Index (from July 2008)')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()
