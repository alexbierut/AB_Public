import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CPI data
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

# Calculate the monthly inflation rate using percentage change
data['Inflation_Rate_Percentage'] = data['CPI'].pct_change() * 100

# Calculate the monthly BER assuming it can be derived from the inflation rate
# Convert the percentage rate to a rate
data['Inflation_Rate_Rate'] = data['Inflation_Rate_Percentage'] / 100

# Assuming that the BER_yearly is equivalent to the monthly inflation rate extrapolated to a year
data['BER_yearly'] = (1 + data['Inflation_Rate_Rate'])**12 - 1

# Deannualize the BER to get the monthly BER
data['BER_monthly'] = (1 + data['BER_yearly'])**(1/12) - 1

# Extract the monthly inflation rate for February 2013
feb_2013_inflation_rate = data.loc[data['YearMonth'] == '2013-02', 'BER_monthly'].values[0] * 100  # Convert back to percentage

print(f'Monthly Inflation Rate from BER for February 2013: {feb_2013_inflation_rate:.2f}%')

# Plot the monthly inflation rate from BER
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['BER_monthly'] * 100, marker='o', label='Monthly Inflation Rate')
plt.title('Monthly Inflation Rate from BER')
plt.xlabel('Date')
plt.ylabel('Inflation Rate (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
