from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

ps_cpi = pd.read_csv('CPI.csv')
ps_cpi['date'] =pd.to_datetime(ps_cpi['date'])
ps_cpi['YearMonth'] = ps_cpi['date'].dt.strftime('%Y-%m')
cpi =ps_cpi.drop_duplicates('YearMonth', keep='last').copy().reset_index().drop(['index'],axis=1)
cpi_train = (cpi[cpi.YearMonth<'2013-09'].copy())
cpi_test = (cpi[cpi.YearMonth >='2013-09'].copy())
model = LinearRegression().fit(np.array(cpi_train.index).reshape(-1, 1),cpi_train.CPI)
coefficients = [model.coef_[0],model.intercept_]
print("The linear trend is given by F(t) = " +str(coefficients[0])+"*t + (" +str(coefficients[1])+")")
linear_cpi =model.predict(np.array(cpi_train.index).reshape(-1, 1))
#x = input('Choose Method:')
x= "linear"
if x == 'linear':
    remaining = cpi_train.CPI - linear_cpi
    linear_cpi_test = model.predict(np.array(cpi_test.index).reshape(-1, 1))
    remaining_test = cpi_test.CPI - linear_cpi_test
    test = cpi_test.index
    train = cpi_train.index
else:
    cpi_diff_log = np.log(cpi.CPI).diff()
    remaining = cpi_diff_log[0:cpi_train.shape[0]]
    remaining.iloc[0] = 0
    #.reset_index()#.drop(['index'],axis =1)
    remaining_test = cpi_diff_log[cpi_train.shape[0]:].dropna()
    remaining_test.iloc[0] = 0
    test = cpi_test.index
    train = cpi_train.index
def rebuild_diffed(series, first_element_original,x,linear_trend):
    if x == 'linear':
        final = series + linear_trend
    else:
        cumsum = pd.Series(series).cumsum()
        final = np.exp(cumsum.fillna(0) + first_element_original)
        if first_element_original == 0:
            final = np.exp(cumsum.fillna(0))
    return final


n = 2
AR2_model = AutoReg(remaining, lags= n).fit()# Here we have used the default value for the trend parameter
coef = AR2_model.params
print(coef)

# walk forward over time steps in test
past = remaining[len(remaining)-n:].values
past = [past[i] for i in range(len(past))]
test_predictions = list()
training_predictions = AR2_model.predict(start = train[0], end =train[-1])
for t in range(len(remaining_test)):
    length = len(past)
    lag = [past[i] for i in range(length-n,length)]
    pred = coef[0]
    for d in range(n):
        pred += coef[d+1] * lag[n-d-1]
    obs = remaining_test.values[t]
    test_predictions.append(pred)
    past.append(obs)
final_training = (rebuild_diffed(training_predictions,np.log(cpi_train.CPI[0]),x,linear_cpi))
final_test = (rebuild_diffed(test_predictions,np.log(cpi_train.iloc[-1].CPI),x,linear_cpi_test))
residuals = (remaining - training_predictions)[2:]
rmse = mean_squared_error(cpi_test.dropna().CPI,final_test[:-2])**0.5
print("The rmse of the final fit is " + str(rmse))