from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg

import statsmodels.api as sm
import statsmodels.stats.diagnostic

import numpy
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


path    = os.path.split(os.path.realpath(__file__))[0]
apath = os.path.join(path,'../')
if apath not in sys.path: sys.path.append(apath)

import test_baostock
import stock_day_database as db

IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./images")
def test():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000019")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)

    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['close']

    ## origin 
    plt.figure()
    plt.plot(hp_ret)
    plt.savefig(os.path.join(IMAGE_PATH,"origin.png"))
    ## ARIMA
    plt.figure()
    plot_acf(hp_ret)
    plt.savefig(os.path.join(IMAGE_PATH,"acf.png"))
     
    plt.figure()
    plot_pacf(hp_ret)
    plt.savefig(os.path.join(IMAGE_PATH,"pacf.png"))
    #平稳性检测
    print(u'原始序列的ADF检验结果为：', ADF(hp_ret))
    #返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

    model = auto_arima(hp_ret, seasonal_order=(15,1,3),stationary=True, information_criterion='aic', seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    print(model.summary())

    print(model.conf_int())

    # # Forecast
    # n_periods = 24
    # fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    # index_of_fc = numpy.arange(len(data['Average_House_Price']), len(data['Average_House_Price'])+n_periods)

    # print(fc)
    # print(confint)
    # # make series for plotting purpose
    # fc_series = pandas.Series(fc, index=index_of_fc)
    # lower_series = pandas.Series(confint[:, 0], index=index_of_fc)
    # upper_series = pandas.Series(confint[:, 1], index=index_of_fc)

    # # Plot
    # plt.figure()
    # plt.plot(data['Average_House_Price'])
    # # plt.plot(fc_series, color='darkgreen')
    # # plt.fill_between(lower_series.index, 
    # #                 lower_series, 
    # #                 upper_series, 
    # #                 color='k', alpha=.15)

    # plt.title("Final Forecast of UKHP")
    # # plt.show()
    # plt.savefig(os.path.join(images_path,"forecast.png"))

    ### ------------------
    ### -------------------------------
    model_fit = AutoReg(hp_ret,lags=[1,11,14]).fit()
    params = model_fit.params
    # p = model_fit.k_ar  # 即时间序列模型中常见的p，即AR(p), ARMA(p,q), ARIMA(p,d,q)中的p。
    # p的实际含义，此处得到p=29，意味着当天的温度由最近29天的温度来预测。

    model_fit.summary()
    print(model_fit)
    print(params)
    # print(p)

    

if __name__=="__main__":
    test()
