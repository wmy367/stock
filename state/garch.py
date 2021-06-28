from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.arima.model import ARIMA
# from pmdarima import auto_arima
from pmdarima.arima import auto_arima
from statsmodels.tsa.ar_model import AutoReg

import statsmodels.api as sm
import statsmodels.stats.diagnostic

from statsmodels.tsa.stattools import adfuller #ADF单位根检验
from statsmodels.tsa import stattools #白噪声检验:Ljung-Box检验

import numpy
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

from arch import arch_model
import warnings

from pandas.plotting import lag_plot

## 用于时间格式转换
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

path    = os.path.split(os.path.realpath(__file__))[0]
apath = os.path.join(path,'../')
if apath not in sys.path: sys.path.append(apath)

import test_baostock
import stock_day_database as db

IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./images")

import contextlib
@contextlib.contextmanager
def new_png(name):
    plt.figure()
    yield
    plt.savefig(os.path.join(IMAGE_PATH,name))

def test():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000019")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['close'].astype(float)

    print(hp_ret.head())
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['ubuntu mono'] #字体为黑体
    plt.rcParams['axes.unicode_minus'] = False #正常显示负号 #时序图的绘制
    plt.plot(hp_ret)
    plt.xticks(rotation=45) #坐标角度旋转
    plt.xlabel('日期') #横、纵坐标以及标题命名
    plt.ylabel('开盘价')
    plt.title('中国联通开盘价',loc='center')
    plt.savefig(os.path.join(IMAGE_PATH,"origin.png"))

    ts = hp_ret

    result = adfuller(ts) #不能拒绝原假设，即原序列存在单位根
    print(result)   

    ts1= ts.diff().dropna() #一阶差分再进行ADF检验
    result = adfuller(ts1)
    print(result)
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['ubuntu mono'] #字体为黑体
    plt.rcParams['axes.unicode_minus'] = False #正常显示负号
    plt.xticks(rotation=45) #坐标角度旋转
    plt.xlabel('日期') #横、纵坐标以及标题命名
    plt.ylabel('开盘价')
    plt.title('差分后的开盘价',loc='center')
    plt.plot(ts1)
    # plt.show() #一阶差分后的时序图
    plt.savefig(os.path.join(IMAGE_PATH,"diff_1.png"))


    LjungBox=stattools.q_stat(stattools.acf(ts1)[1:12],len(ts1))[1] #显示第一个到第11个白噪声检验的p值
    print(LjungBox)  #检验的p值大于0.05，因此不能拒绝原假设，差分后序列白噪声检验通过  

    model=ARIMA(ts,order=(1,1,0)) #白噪声检验通过，直接确定模型
    result=model.fit(disp=-1)
    print(result.summary()) #提取模型信息

    with new_png("acf.png") as f:
        plot_acf(ts1,use_vlines=True,lags=30) #自相关函数图，滞后30阶

    with new_png('pacf.png') as f:
        plot_pacf(ts1,use_vlines=True,lags=30) #偏自相关函数图

    # train_results = sm.tsa.arma_order_select_ic(ts1, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
    # print('AIC', train_results.aic_min_order) #建立AIC值最小的模型
    # print('BIC', train_results.bic_min_order)
    model = ARIMA(ts,(5,1,4)).fit()
    print(model.summary()) #提取模型系数等信息，保留三位小数；summary2保留四位小数

    ##模型诊断

    print(model.conf_int()) #系数显著性检验

    stdresid=model.resid/math.sqrt(model.sigma2) #标准化残差
    with new_png('标准化残差序列图.png') as f:
        plt.plot(stdresid)
        plt.xticks(rotation=45) #坐标角度旋转
        plt.xlabel('日期') #横、纵坐标以及标题命名
        plt.ylabel('标准化残差')
        plt.title('标准化残差序列图',loc='center')

    with new_png('stdresid_acf.png'):
        plot_acf(stdresid,lags=30) 

    LjungBox=stattools.q_stat(stattools.acf(stdresid)[1:13],len(stdresid))
    print(LjungBox)
    # LjungBox[1][-1] #LjungBox检验的最后一个P值，大于0.05，通过白噪声检验

    ## 模型预测
    a=model.forecast(5)
    print(a)
    
    with new_png('forecast.png'):
        fig, ax = plt.subplots(figsize=(6, 4))
        # ts.plot()
        ax = ts.loc['2021-04':].plot(ax=ax)
        # plt.plot(ts.loc['2021-04':])
    
    with new_png('plot_predict.png'):
        # fig = model.plot_predict(end="2021-06-01")
        fig = model.plot_predict(5,len(ts)+10)

    ## ARCH模型

    resid1=result.resid #提取残差
    LjungBox=stattools.q_stat(stattools.acf(resid1**2)[1:13],len(resid1)) #残差平方序列的白噪声检验
    print(LjungBox)
    # LjungBox[1][-1] #拒绝原假设，则残差序列具有ARCH效应

    am=arch_model(resid1) #默认模型为GARCH（1，1）
    model2=am.fit(update_freq=0) #估计参数

    print(model2.summary())

def test_other():
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)
    # warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)

    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000019")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['close'].astype(float)

    print(hp_ret.head())
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['ubuntu mono'] #字体为黑体
    plt.rcParams['axes.unicode_minus'] = False #正常显示负号 #时序图的绘制
    plt.plot(hp_ret)
    plt.xticks(rotation=45) #坐标角度旋转
    plt.xlabel('日期') #横、纵坐标以及标题命名
    plt.ylabel('开盘价')
    plt.title('中国联通开盘价',loc='center')
    plt.savefig(os.path.join(IMAGE_PATH,"origin.png"))

    ts = hp_ret

    ## lag 相关检测
    with new_png('lag.png'):
        lag_plot(ts,lag=1)   # 默认lag=1

      ## ARIMA
    with new_png('acf.png'):
        plot_acf(hp_ret)
     
    with new_png('pacf.png'):
        plot_pacf(hp_ret)

    #平稳性检测
    print(u'原始序列的ADF检验结果为：', ADF(hp_ret)) #存在单位根的可能性
    #返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
    # 不能拒绝原假设，即原序列存在单位根

    # model = auto_arima(hp_ret, seasonal_order=(15,2,3),stationary=True, information_criterion='aic', seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    # model = auto_arima(hp_ret.dropna(),start_p=1, d=1, start_q=1,stationary=True, information_criterion='aic', seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

    # model = auto_arima(hp_ret,start_p=0, d=None, start_q=0, max_p=5, max_d=2, max_q=5, max_order=20, m=1, seasonal=True, information_criterion='aic', trend=True, uppress_warnings=True, error_action='ignore', trace=False,with_intercept='auto')
    model = auto_arima(hp_ret,trace=True, uppress_warnings=True, error_action='ignore',information_criterion='bic')
    print('summary',model.summary())

    print('conf',model.conf_int())

    ## AUTO ARMMA 似乎不好使用，所以使用test的方法

    ts1 = ts.diff().dropna()
    train_results = sm.tsa.arma_order_select_ic(ts1, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)
    # train_results = sm.tsa.stattools.arma_order_select_ic(ts1, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)
    print('AIC', train_results.aic_min_order) #建立AIC值最小的模型
    print('BIC', train_results.bic_min_order) #建立BIC值最小的模型

    model = ARIMA(ts,(3,1,2)).fit()
    print(model.summary()) #提取模型系数等信息，保留三位小数；summary2保留四位小数
    ##模型诊断
    print(model.conf_int()) #系数显著性检验

    stdresid=model.resid/math.sqrt(model.sigma2) #标准化残差
    with new_png('标准化残差序列图.png') as f:
        plt.plot(stdresid)
        plt.xticks(rotation=45) #坐标角度旋转
        plt.xlabel('日期') #横、纵坐标以及标题命名
        plt.ylabel('标准化残差')
        plt.title('标准化残差序列图',loc='center')

    with new_png('stdresid_acf.png'):
        plot_acf(stdresid,lags=30) 

    LjungBox=stattools.q_stat(stattools.acf(stdresid)[1:13],len(stdresid))
    print(LjungBox)
    # LjungBox[1][-1] #LjungBox检验的最后一个P值，大于0.05，通过白噪声检验

    ## 模型预测
    a=model.forecast(5)
    print(a)
    
    with new_png('forecast.png'):
        fig, ax = plt.subplots(figsize=(6, 4))
        # ts.plot()
        ax = ts.loc['2021-04':].plot(ax=ax)
        # plt.plot(ts.loc['2021-04':])
    
    with new_png('plot_predict.png'):
        # fig = model.plot_predict(end="2021-06-01")
        fig = model.plot_predict(5,len(ts)+10)

    ## ARCH模型

    resid1=model.resid #提取残差
    LjungBox=stattools.q_stat(stattools.acf(resid1**2)[1:13],len(resid1)) #残差平方序列的白噪声检验
    print('LjungBox\n',LjungBox)
    # LjungBox[1][-1] #拒绝原假设，则残差序列具有ARCH效应

    am=arch_model(resid1) #默认模型为GARCH（1，1）
    model2=am.fit(update_freq=0) #估计参数

    print('summary\n',model2.summary())

if __name__ == "__main__":
    # test()

    test_other()