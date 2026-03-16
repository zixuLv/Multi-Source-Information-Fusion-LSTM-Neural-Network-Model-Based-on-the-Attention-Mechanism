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

# 读取数据
#导入不同窗口期数据，同时修改策略训练窗口长度
data = pd.read_csv('/home/zyh/pystudy/baseline-18.csv', encoding='utf-8')


# 将时间作为索引
data.Timestamp = pd.to_datetime(data.Timestamp)
data.index = data.Timestamp
#取对数做差分
data_rate1=np.log1p(data['Weighted_Price']).diff(1)
data_rate2=np.log1p(data['Weighted_Price'])
data_rate = {
    "Timestamp":data.index[1:],  
    "Weighted_Price_rate":data_rate1[1:]
}
df = pd.DataFrame(data_rate)
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
data_rate=df.set_index(['Timestamp'], drop=True)

#ADF检验结果
d = 0
adf=ADF(data_rate['Weighted_Price_rate'])
p=adf[1]
while p >=0.1:
    print("ADF检验结果不能拒绝原假设，数据不平稳")
    d = d + 1
    adf = ADF(data['Weighted_Price_rate'].diff(d).dropna())
    p=adf[1]
print('对数据进行差分直至平稳，差分d次数为{}，p值为{}'.format(d, p))    

#白噪声检验
acorr=acorr_ljungbox(data_rate['Weighted_Price_rate'].diff().dropna(), lags=1)
print(acorr)
a=acorr.lb_pvalue[1]
print(a)
if a >= 0.1:
    print("差分后白噪声检验不能拒绝原假设，时间序列为白噪声")
    print("不能建立模型")
else:
    print("时间序列不是白噪声，可以建立模型")

#正态性检验
ks = kstest(data_rate['Weighted_Price_rate'],cdf = "norm")
ksp = ks[1]
print(ksp)
if ksp > 0.1:
    print("收益率数据与正态分布没有显著差异")
else:
    print("收益率数据不符合正态分布")
sns.distplot(data_rate['Weighted_Price_rate'])


#日度预测
df_daily = data_rate[['Weighted_Price_rate']]
df_daily['Weighted_Price'] = data[['Weighted_Price']]
df_daily['Timestamp'] = data[['Timestamp']]

x = data_rate['Weighted_Price_rate']
#设置训练窗口长度
start_loc = 0
end_loc = 18
train = x[:end_loc]
test = x[end_loc:]
df_correct=df_daily[end_loc-1:]

#AR模型
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

#ARMA模型
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

#ARMA模型
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

#Garch模型
#AR(1)-Garch模型
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
"""forecast默认得到的值为对下一天的预测值，例如8月31号索引对应值即为对9月1号预测值,因此加shift(1)手动调整"""
df_correct['AR1Garchforecast']=pd.DataFrame(forecasts).T['h.1'].shift(1)


#AR(2)-Garch模型
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

#AR(1)-EGarch模型
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

#AR(2)-EGarch模型
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

#AR(1)-arch模型
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

#AR(2)-arch模型
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
#MAPE需要自己实现
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
    
#重置索引
# df_correct=df_correct.reset_index(drop=True)
# print(df_correct.index.values)
# exit()

#预测涨跌准确率
def accuracy(m1,u1):
    p=0
    q=0
    z = 0
    d = 0
    z1 = 0
    d1 = 0
    e=0
    for ar1 in df_correct['Weighted_Price_rate']:
        if ar1 != 0:
            ind = list(df_correct['Weighted_Price_rate']).index(ar1)
            #print(ind)
            ar1_pre=m1[ind]
            if ar1*ar1_pre>0:
                p=p+1
                print("预测涨跌准确",p)
                
                if ar1 >0 and ar1_pre >0:
                    z = z+1
                if ar1 <0 and ar1_pre <0:
                    d = d+1
            else:
                q=q+1
                print("预测涨跌不准确",q)
                
                if ar1 >0 and  ar1_pre <0:
                    z1 = z1+1
                if ar1 <0 and  ar1_pre >0:
                    d1 = d1+1
        else:
            e = e+1
            print("剔除实际收益率为0的情况")

    print("实际上涨次数",z+z1,"预测正确次数",z,"实际下跌次数",d+d1,"预测正确次数",d,"收益率为0次数",e)
    print("预测上涨次数",z+d1,"预测下跌次数",d+z1)
    print("预测上涨正确率",z/(z+d1),"预测下跌正确率",d/(d+z1))
    
    return p/(p+q)
if __name__=="__main__":
    
    print('AR(1)模型预测涨跌准确率',accuracy(df_correct['AR1forecast'],'AR(1)'))
    print('AR(2)模型预测涨跌准确率',accuracy(df_correct['AR2forecast'],'AR(2)'))
    print('ARMA(1,1)模型预测涨跌准确率',accuracy(df_correct.ARMA11forecast,'ARMA(1,1)'))
    print('ARMA(2,2)模型预测涨跌准确率',accuracy(df_correct.ARMA22forecast,'ARMA(2,2)'))
    print('AR(1)-Garch(1,1)模型预测涨跌准确率',accuracy(df_correct.AR1Garchforecast,'AR(1)-Garch(1,1)'))
    print('AR(2)-Garch(1,1)模型预测涨跌准确率',accuracy(df_correct.AR2Garchforecast,'AR(2)-Garch(1,1)'))
    print('AR(1)-EGarch(1,1)模型预测结果',accuracy(df_correct.AR1EGarchforecast,'AR(1)-EGarch(1,1)'))
    print('AR(2)-EGarch(1,1)模型预测结果',accuracy(df_correct.AR2EGarchforecast,'AR(2)-EGarch(1,1)'))
    print('AR(1)-Arch(1,1)模型预测结果',accuracy(df_correct.AR1Archforecast,'AR(1)-Arch(1,1)'))
    print('AR(2)-Arch(1,1)模型预测结果',accuracy(df_correct.AR2Archforecast,'AR(2)-Arch(1,1)'))
    


date_list=[]
for i in list(df_correct.index.values):
    i=str(i)
    i=i.split('T')[0]
    date_list.append(i)
count = len(data_rate["Weighted_Price_rate"])+1

def save_stratege(strategy,i):
    with open("BTC_the_strategy_%s.txt"%i,"w",encoding="utf-8")as f:
        for stock,actions in strategy.items():
            f.write("%s\t%s\n"%(stock,actions))

#计算收益
def get_date_range2(begin_date, end_date, freq=1, format='%Y-%m-%d', include_end=True):
    """
    获取指定日期内的所有日期，可指定周期, 至少返回两条日期
    :param begin_date: 起始日期
    :param end_date: 结束日期
    :param freq: 时间间隔
    :param format: 格式化输出
    :param include_end: 是否包含最后日期，为False则计算到end_date前一天的日期
    :return: 
    """
    date_list = []
    now = datetime.datetime.now()
    begin_ = now + datetime.timedelta(days=-freq)
    end_ = now.strftime(format)
    if not begin_date and not end_date:
        # 如果begin_date和end_date都为None，则返回今天和上一个周期的日期
        return [begin_, end_]
    if not begin_date:
        # 如果没有起始日期，获取今天之前一个周期的日期
        begin_date = begin_
    if not end_date:
        # 如果没有结束日期，获取今天的日期
        end_date = end_
    begin_ = begin_date if isinstance(begin_date, datetime.datetime) else datetime.datetime.strptime(str(begin_date), format)
    end_ = end_date if isinstance(end_date, datetime.datetime) else datetime.datetime.strptime(str(end_date), format)
    if begin_ >= end_:
        # 如果起始日期大于结束日期，起始日期改为上一个周期
        begin_ = end_ + datetime.timedelta(days=-freq)
    if not include_end:
        # 如果不包含结束日期，获取end的前一天
        end_ = end_ + datetime.timedelta(days=-1)
    while begin_ < end_:
        # 遍历起始和结束的日期直到起始日期大于等于end结束
        if format:
            date_str = begin_.strftime(format)
            date_list.append(date_str)
        else:
            date_list.append(begin_)
        begin_ = begin_ + datetime.timedelta(days=freq)
    # 添加结束日期
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

#运行时需要手动隐藏另一个策略代码
#没有阈值的买入持有策略
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
                #判断是否已经买入,没有则买入
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
    a=["AR1forecast","AR2forecast","ARMA11forecast","ARMA22forecast","AR1Garchforecast","AR2Garchforecast",
        "AR1Archforecast","AR2Archforecast"]
    for i in a:
        strategy=_return(df_correct[i],date_list,i)
        save_stratege(strategy,i)
        

#设置阈值的买入持有策略
#阈值为交易日前20日平均对数收益率
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
            rate = data_rate["Weighted_Price_rate"]

            limit = sum(rate[9+n:29+n])
            
            if pre2 > 0 and pre2>limit and buy_num==0:
                #判断是否已经买入,没有则买入
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
    

print("运行结束")
