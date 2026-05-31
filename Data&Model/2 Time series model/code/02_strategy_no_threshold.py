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

# Read data
#Import data from different time periods and different windows, while adjusting the strategy training window length（end_loc）
data = pd.read_csv('D:/btc data from different time windows/baseline-18.csv', encoding='utf-8')

# Use time as an index
data.Timestamp = pd.to_datetime(data.Timestamp)
data.index = data.Timestamp

data_rate1=np.log1p(data['Weighted_Price']).diff(1)
data_rate2=np.log1p(data['Weighted_Price'])
data_rate = {
    "Timestamp":data.index[1:],  
    "Weighted_Price_rate":data_rate1[1:]
}
df = pd.DataFrame(data_rate)
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
data_rate=df.set_index(['Timestamp'], drop=True)

#Daily Forecast
df_daily = data_rate[['Weighted_Price_rate']]
df_daily['Weighted_Price'] = data[['Weighted_Price']]
df_daily['Timestamp'] = data[['Timestamp']]

x = data_rate['Weighted_Price_rate']
#Set the training window length
start_loc = 0
end_loc = 18
train = x[:end_loc]
test = x[end_loc:]
df_correct=df_daily[end_loc-1:]

#AR model
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

#ARMA model
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

#Garch model
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


date_list=[]
for i in list(df_correct.index.values):
    i=str(i)
    i=i.split('T')[0]
    date_list.append(i)
count = len(data_rate["Weighted_Price_rate"])+1
print(count)
def save_stratege(strategy,i):
    with open("no_threshold_%s.txt"%i,"w",encoding="utf-8")as f:
        for stock,actions in strategy.items():
            f.write("%s\t%s\n"%(stock,actions))


def get_date_range2(begin_date, end_date, freq=1, format='%Y-%m-%d', include_end=True):
    
    date_list = []
    now = datetime.datetime.now()
    begin_ = now + datetime.timedelta(days=-freq)
    end_ = now.strftime(format)
    if not begin_date and not end_date:
        # If both `begin_date` and `end_date` are `None`, the function returns today's date and the date of the previous period.
        return [begin_, end_]
    if not begin_date:
        # If there is no start date, return the date of the previous cycle
        begin_date = begin_
    if not end_date:
        # If there is no end date, retrieve today's date
        end_date = end_
    begin_ = begin_date if isinstance(begin_date, datetime.datetime) else datetime.datetime.strptime(str(begin_date), format)
    end_ = end_date if isinstance(end_date, datetime.datetime) else datetime.datetime.strptime(str(end_date), format)
    if begin_ >= end_:
        # If the start date is later than the end date, change the start date to the previous period.
        begin_ = end_ + datetime.timedelta(days=-freq)
    if not include_end:
        # If no end date is specified, return the day before the end date
        end_ = end_ + datetime.timedelta(days=-1)
    while begin_ < end_:
        # Iterate through the start and end dates until the start date is greater than or equal to the end date.
        if format:
            date_str = begin_.strftime(format)
            date_list.append(date_str)
        else:
            date_list.append(begin_)
        begin_ = begin_ + datetime.timedelta(days=freq)
    # Add an end date
    if format:
        date_list.append(end_.strftime(format))
    else:
        date_list.append(end_)
    return date_list
def date2num(date):
    try:
        year,month,day=[int(x)for x in date.split("-")]
    except:
        print("Please check the date. E.g., 2015-9-10")
        exit()
    ret=year*10000+month*100+day
    return ret

#A buy-and-hold strategy without a threshold
def _return(column,date_list,i):
    buy_num=0
    strategy={}
    strategy['btc']=[]
    up_down={}
    up_down['i']=[]
    n=0
    for pre in column:
        s=[]
        p=[]
        n=n+1
        if n > count-end_loc:
            break
        elif n < count-end_loc:
            pre2 = column[n]
            if pre2 > 0 and buy_num==0:
                #Check if the position has already been opened; if not, open it.
                buy_num = 1
                p.append(date2num(date_list[n-1]))
                p.append(1)
                s.append(date2num(date_list[n-1]))
                s.append('buy')
                s.append(1)
                strategy['btc'].append(s)
                up_down['i'].append(p)
            elif pre2 <0 and buy_num!=0:
                p.append(date2num(date_list[n-1]))
                p.append(0)
                s.append(date2num(date_list[n-1]))
                s.append('sell')
                s.append(buy_num)
                up_down['i'].append(p)
                strategy['btc'].append(s)
                buy_num=0
                continue
            elif buy_num==0:
                s.append(date2num(date_list[n-1]))
                s.append('none')
                s.append(0)
                strategy['btc'].append(s)
            elif buy_num!=0:
                s.append(date2num(date_list[n-1]))
                s.append('keep')
                s.append(buy_num)
                strategy['btc'].append(s)
        elif n == count-end_loc:
            if buy_num!=0:
                p.append(date2num(date_list[n-1]))
                p.append(1)
                s.append(date2num(date_list[n-1]))
                s.append('sell')
                s.append(buy_num)
                up_down['i'].append(p)
                strategy['btc'].append(s)
            if buy_num==0:
                s.append(date2num(date_list[n-1]))
                s.append('none')
                s.append(0)
                strategy['btc'].append(s)
    return strategy
            
    
if __name__=="__main__":
    a=["AR1forecast","AR2forecast","ARMA11forecast","ARMA22forecast","AR1Garchforecast","AR2Garchforecast","AR1Archforecast","AR2Archforecast"]
    for i in a:
        strategy=_return(df_correct[i],date_list,i)
        save_stratege(strategy,i)

print("done")
