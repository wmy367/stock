from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

import statsmodels.api as sm
import statsmodels.stats.diagnostic

import numpy
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import lag_plot


path    = os.path.split(os.path.realpath(__file__))[0]
apath = os.path.join(path,'../')
if apath not in sys.path: sys.path.append(apath)

import test_baostock
import stock_day_database as db

IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./images")

def test():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000004")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)

    dd = numpy.abs(df['pctChg'])
    df['xl'] = (dd>=5)*4 + (dd>=3)*(dd<5)*3 + (dd>=1.5)*(dd<3)*2 + (dd>=0.5)*(dd<1.5)*1 + (dd<0.5)*0
    df['xl'] = df['xl']*(df['pctChg']<0)*(-1) + df['xl']*(df['pctChg']>=0)*1
    df['xl'] = df['xl'] / 4
    

    print(df['xl'])

    # tdata = numpy.log(df['pctChg']/100 + 1) * 100
    # tdata = df['close']
    # tdata = df['pctChg']
    # tdata = numpy.asarray(list(df['volume']))
    # tdata = (tdata[1:] - tdata[:-1]) / tdata[:-1]
    # tdata = pd.Series(tdata)
    # tdata = numpy.log(tdata + 1)
    # tdata = df['volume']
    # tdata = (tdata.shift() - tdata ) / tdata
    # tdata = numpy.log( tdata[1:] + 1)

    tdata = df['xl']

    # print(df['pctChg'])
    # print(df['volume'][:-1] ,df['volume'][1:])
    # print(tdata)

    ## origin 
    plt.figure()
    plt.plot(numpy.log(df['pctChg']/100 + 1) * 1)
    plt.plot(df['volume']/numpy.max(df['volume']))
    plt.plot(df['amount']/numpy.max(df['amount']))
    plt.plot(tdata)
    
    plt.savefig(os.path.join(IMAGE_PATH,"origin.png"))

    ## lag 相关检测
    plt.figure()
    lag_plot(tdata,lag=1)   # 默认lag=1
    plt.savefig(os.path.join(IMAGE_PATH,"lag.png"))

    ## 平稳性检测
    dftest = adfuller(tdata, autolag='AIC')
    print(dftest)
    ## 随机性检测
    p_value = acorr_ljungbox(tdata, lags=20,return_df=True) #lags可自定义
    print(p_value)


if __name__=="__main__":
    test()