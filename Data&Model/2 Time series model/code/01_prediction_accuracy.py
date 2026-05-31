# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 13:10:54 2022

@author: zyh
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools
import datetime
#from statsmodels.tsa.arima_model import ARMA
from datetime import datetime ,timedelta, date
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa import stattools
from statsmodels.tsa.ar_model import AutoReg
import arch 
import numpy as np
import math
from sklearn import metrics
from scipy.stats import kstest
import seaborn as sns
import sys
from arch.univariate import GARCH

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#  Read data
#Import data from different time periods and different windows, while adjusting the strategy training window length（end_loc）
data = pd.read_csv('D:/btc data from different time windows/baseline-18.csv', encoding='utf-8')

#Use time as an index
data.Timestamp = pd.to_datetime(data.Timestamp)
data.index = data.Timestamp

data_rate1=np.log1p(data['Weighted_Price']).diff(1)
data_rate2=np.log1p(data['Weighted_Price'])
data_rate = {
    "Timestamp":data.index[1:],  
    "Weighted_Price_rate":data_rate1[1:]}
df = pd.DataFrame(data_rate)
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
data_rate=df.set_index(['Timestamp'], drop=True)

#ADF
d = 0
adf=ADF(data_rate['Weighted_Price_rate'])
p=adf[1]
while p >=0.1:
    print("The ADF test results do not reject the null hypothesis; the data are non-stationary.")
    d = d + 1
    adf = ADF(data['Weighted_Price_rate'].diff(d).dropna())
    p=adf[1]
print('Differentiate the data until it is smooth; the number of differentiations is{}，p is{}'.format(d, p))    

#White Noise Test
acorr=acorr_ljungbox(data_rate['Weighted_Price_rate'].diff().dropna(), lags=1)
print(acorr)
a=acorr.lb_pvalue[1]
print(a)
if a >= 0.1:
    print("After differencing, the white noise test fails to reject the null hypothesis; the time series is white noise.")
    print("Unable to create a model")
else:
    print("Time series are not white noise; they can be modeled.")

#Normality Test
ks = kstest(data_rate['Weighted_Price_rate'],cdf = "norm")
ksp = ks[1]
print(ksp)
if ksp > 0.1:
    print("The yield data show no significant deviation from a normal distribution.")
else:
    print("The yield data does not follow a normal distribution")
sns.distplot(data_rate['Weighted_Price_rate'])

#Daily Forecast
df_daily = data_rate[['Weighted_Price_rate']]
#date_list =[]
df_daily['Weighted_Price'] = data[['Weighted_Price']]
df_daily['Timestamp'] = data[['Timestamp']]

x = data_rate['Weighted_Price_rate']
#Set the training window length
start_loc = 0
end_loc = 18
train = x[:end_loc]
test = x[end_loc:]
df_correct=df_daily[end_loc-1:]

#AR
#AR(1)
forecasts = {}
for i in range(len(test)):
    train = x[i:end_loc+i]
    ar = AutoReg(train, 1).fit()   
    t = ar.forecast(steps=1)
    t = list(t.items())
    forecasts.update(t)
print()
df_correct['AR1forecast']=pd.DataFrame(forecasts,index=[0]).T

#AR(2)
forecasts = {}
for i in range(len(test)):
    train = x[i:end_loc+i]
    ar = AutoReg(train, 2).fit()   
    t = ar.forecast(steps=1)
    t = list(t.items())
    forecasts.update(t)
print()
df_correct['AR2forecast']=pd.DataFrame(forecasts,index=[0]).T

#ARMA
#ARMA(1,1)
forecasts = {}
for i in range(len(test)):
    train = x[i:end_loc+i]
    arma = ARIMA(train, order=(1,0,1)).fit()   
    t = arma.forecast(steps=1)
    t = list(t.items())
    forecasts.update(t)
print()
df_correct['ARMA11forecast']=pd.DataFrame(forecasts,index=[0]).T

#ARMA
#ARMA(2,2)
forecasts = {}
for i in range(len(test)):
    train = x[i:end_loc+i]
    arma = ARIMA(train, order=(2,0,2)).fit()   
    t = arma.forecast(steps=1)
    t = list(t.items())
    forecasts.update(t)
print()
df_correct['ARMA22forecast']=pd.DataFrame(forecasts,index=[0]).T

#Garch
#AR(1)-Garch
am = arch.arch_model(x, mean='AR', lags=1, vol='garch',dist="StudentsT")
forecasts = {}
for i in range(len(test)+1):
    train = x[i:end_loc+i]
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    t = res.forecast(horizon=1)
    temp = t.mean
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
"""By default, the `forecast` function returns the forecast value for the following day. 
For example, the value associated with the index 31 August is the forecast for 1 September; therefore, use `shift(1)` to manually adjust it."""
df_correct['AR1Garchforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)

#AR(2)-Garch
am = arch.arch_model(x, mean='AR', lags=2, vol='garch',dist="StudentsT")
forecasts = {}
for i in range(len(test)+1):
    train = x[i:end_loc+i]
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    t = res.forecast(horizon=1)
    temp = t.mean
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
df_correct['AR2Garchforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)


#AR(1)-EGarch
am = arch.arch_model(x, mean='AR', lags=1, vol='egarch',dist="StudentsT")
forecasts = {}
for i in range(len(test)+1):
    train = x[i:end_loc+i]
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    t = res.forecast(horizon=1)
    temp = t.mean
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
df_correct['AR1EGarchforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)

#AR(2)-EGarch
am = arch.arch_model(x, mean='AR', lags=2, vol='egarch',dist="StudentsT")
forecasts = {}
for i in range(len(test)+1):
    train = x[i:end_loc+i]
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    t = res.forecast(horizon=1)
    temp = t.mean
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
df_correct['AR2EGarchforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)


#AR(1)-arch
am = arch.arch_model(x, mean='AR', lags=1, vol='arch',dist="StudentsT")
forecasts = {}
for i in range(len(test)+1):
    train = x[i:end_loc+i]
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    t = res.forecast(horizon=1)
    temp = t.mean
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
df_correct['AR1Archforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)

#AR(2)-arch
am = arch.arch_model(x, mean='AR', lags=2, vol='arch',dist="StudentsT")
forecasts = {}
for i in range(len(test)+1):
    train = x[i:end_loc+i]
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    t = res.forecast(horizon=1)
    temp = t.mean
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
df_correct['AR2Archforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)


#RMSE、MAPE
#MAPE must be implemented manually
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
#AR1
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR1forecast[1:]

print('AR1MSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR1RMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR1MAE:',metrics.mean_absolute_error(y_true, y_pred))

if __name__=="__main__":
    print('AR1MAPE:',mape(y_true, y_pred))

#AR2
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR2forecast[1:]
print('AR2MSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR2RMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR2MAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR2MAPE:',mape(y_true, y_pred))

#ARMA11
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.ARMA11forecast[1:]
print('ARMA11MSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('ARMA11RMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('ARMA11MAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('ARMA11MAPE:',mape(y_true, y_pred))

#ARMA22
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.ARMA22forecast[1:]
print('ARMA22MSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('ARMA22RMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('ARMA22MAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('ARMA22MAPE:',mape(y_true, y_pred))

#AR1GARCH
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR1Garchforecast[1:]
print('AR1GarchMSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR1GarchRMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR1GarchMAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR1GarchMAPE:',mape(y_true, y_pred))

#AR2GARCH
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR2Garchforecast[1:]
print('AR2GarchMSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR2GarchRMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR2GarchMAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR2GarchMAPE:',mape(y_true, y_pred))

#AR1EGARCH
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR1EGarchforecast[1:]
print('AR1EGarchMSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR1EGarchRMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR1EGarchMAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR1EGarchMAPE:',mape(y_true, y_pred))

#AR2EGARCH
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR2EGarchforecast[1:]
print('AR2EGarchMSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR2EGarchRMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR2EGarchMAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR2EGarchMAPE:',mape(y_true, y_pred))

#AR1ARCH
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR1Archforecast[1:]
print('AR1ArchMSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR1ArchRMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR1ArchMAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR1ArchMAPE:',mape(y_true, y_pred))

#AR2ARCH
y_true=df_correct.Weighted_Price_rate[1:]
y_pred=df_correct.AR2Archforecast[1:]
print('AR2ArchMSE:',metrics.mean_squared_error(y_true, y_pred)) 
print('AR2ArchRMSE:',math.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print('AR2ArchMAE:',metrics.mean_absolute_error(y_true, y_pred))
if __name__=="__main__":
    print('AR2ArchMAPE:',mape(y_true, y_pred))

#Accuracy of price prediction
def accuracy(m1,u1):
    p=0
    q=0
    z = 0
    d = 0
    z1 = 0
    d1 = 0
    e=0
    for ar1 in df_correct['Weighted_Price_rate'][1:]:
        if ar1 != 0:
            ind = list(df_correct['Weighted_Price_rate']).index(ar1)
            ar1_pre=m1[ind]
            if ar1*ar1_pre>0:
                p=p+1
                
                if ar1 >0 and ar1_pre >0:
                    z = z+1
                if ar1 <0 and ar1_pre <0:
                    d = d+1
            else:
                q=q+1
                if ar1 >0 and  ar1_pre <0:
                    z1 = z1+1
                if ar1 <0 and  ar1_pre >0:
                    d1 = d1+1
        else:
            e = e+1

    print("actual_number_of_increases",z+z1,"correct_predictions_of_increase",z,"actual_number_of_declines",d+d1,"correct_predictions_of_decline",d,"times_the_yield_0",e)
    print("predicted_number_of_increases",z+d1,"predicted_number_of_drops",d+z1)
    print("accuracy_price_increase_predictions",z/(z+d1),"accuracy_down_trend_predictions",d/(d+z1))
    
    return p/(p+q)
if __name__=="__main__":
    
    print('AR(1)_prediction_accuracy',accuracy(df_correct['AR1forecast'],'AR(1)'))
    print('AR(2)_prediction_accuracy',accuracy(df_correct['AR2forecast'],'AR(2)'))
    print('ARMA(1,1)_prediction_accuracy',accuracy(df_correct.ARMA11forecast,'ARMA(1,1)'))
    print('ARMA(2,2)_prediction_accuracy',accuracy(df_correct.ARMA22forecast,'ARMA(2,2)'))
    print('AR(1)-Garch(1,1)_prediction_accuracy',accuracy(df_correct.AR1Garchforecast,'AR(1)-Garch(1,1)'))
    print('AR(2)-Garch(1,1)_prediction_accuracy',accuracy(df_correct.AR2Garchforecast,'AR(2)-Garch(1,1)'))
    print('AR(1)-EGarch(1,1)_prediction_accuracy',accuracy(df_correct.AR1EGarchforecast,'AR(1)-EGarch(1,1)'))
    print('AR(2)-EGarch(1,1)_prediction_accuracy',accuracy(df_correct.AR2EGarchforecast,'AR(2)-EGarch(1,1)'))
    print('AR(1)-Arch(1,1)_prediction_accuracy',accuracy(df_correct.AR1Archforecast,'AR(1)-Arch(1,1)'))
    print('AR(2)-Arch(1,1)_prediction_accuracy',accuracy(df_correct.AR2Archforecast,'AR(2)-Arch(1,1)'))
    
