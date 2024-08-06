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

# Model Training and Prediction
models = [1, 2, 3]  # Polynomial degrees
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
fig.suptitle('CO2 Concentration Models', fontsize=16)

for index, degree in enumerate(models):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train = poly.fit_transform(train_data[['time']])
    y_train = train_data['CO2']
    X_test = poly.transform(test_data[['time']])
    y_test = test_data['CO2']

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics for training
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100

    # Compute metrics for testing
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    # Training data fit and residuals
    axes[0, index].scatter(train_data['time'], y_train, color='blue', label='Actual CO2')
    axes[0, index].plot(train_data['time'], y_train_pred, color='red', label='Predicted CO2')
    axes[1, index].scatter(train_data['time'], y_train - y_train_pred, color='green')
    axes[1, index].axhline(y=0, color='red', linestyle='--')

    # Test data residuals
    axes[2, index].scatter(test_data['time'], y_test - y_test_pred, color='purple')
    axes[2, index].axhline(y=0, color='red', linestyle='--')

    # Set titles and labels
    coef_labels = ' + '.join([f'{coef:.3f}*x^{i}' for i, coef in enumerate(model.coef_, start=1)])
    equation = f'y = {model.intercept_:.3f} + {coef_labels}'
    axes[0, index].set_title(f'Fit: Degree {degree} (Training)')
    axes[1, index].set_title(f'Residuals: Degree {degree} (Training)')
    axes[2, index].set_title(f'Residuals: Degree {degree} (Testing)')
    axes[0, index].legend(title=f'{equation}\nTrain RMSE: {train_rmse:.3f}\nTrain MAPE: {train_mape:.3f}%')
    axes[1, index].legend(title=f'Train RMSE: {train_rmse:.3f}\nTrain MAPE: {train_mape:.3f}%')
    axes[2, index].legend(title=f'Test RMSE: {test_rmse:.3f}\nTest MAPE: {test_mape:.3f}%')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
