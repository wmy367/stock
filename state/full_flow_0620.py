
import os,sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
#tsa为Time Series analysis缩写
# import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.tsatools import unintegrate


import scipy.stats as scs
from arch import arch_model
#画图
import matplotlib.pyplot as plt

#正常显示画图时出现的中文和负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

## 用于时间格式转换
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

path    = os.path.split(os.path.realpath(__file__))[0]
apath = os.path.join(path,'../')
if apath not in sys.path: sys.path.append(apath)

import test_baostock
import stock_day_database as db

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)

IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./FullFlowImage")

import contextlib
@contextlib.contextmanager
def new_png(name):
    plt.figure()
    
    yield
    plt.savefig(os.path.join(IMAGE_PATH,name))

def ts_plot(data, lags=None,title=''):
    if not isinstance(data, pd.Series):   
        data = pd.Series(data)
    #matplotlib官方提供了五种不同的图形风格，
    #包括bmh、ggplot、dark_background、
    #fivethirtyeight和grayscale
    with plt.style.context('ggplot'):    
        fig = plt.figure(figsize=(10, 8))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0))
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        data.plot(ax=ts_ax)
        ts_ax.set_title(title+'时序图')
        plot_acf(data, lags=lags,
                ax=acf_ax, alpha=0.5)
        # plot_acf(data,ax=acf_ax)
        acf_ax.set_title('自相关系数')
        plot_pacf(data, lags=lags,
                ax=pacf_ax, alpha=0.5)
        pacf_ax.set_title('偏自相关系数')
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('QQ 图')        
        scs.probplot(data, sparams=(data.mean(), 
            data.std()), plot=pp_ax)
        pp_ax.set_title('PP 图') 
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_PATH,"%s.png"%title))
    return

def ret_plot(ts, title=''):
    ts1=ts**2
    ts2=np.abs(ts)
    with plt.style.context('ggplot'):    
        fig = plt.figure(figsize=(12, 6))
        layout = (2, 1)
        ts1_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        ts2_ax = plt.subplot2grid(layout, (1, 0))
        ts1.plot(ax=ts1_ax)
        ts1_ax.set_title(title+'日收益率平方')
        ts2.plot(ax=ts2_ax)
        ts2_ax.set_title(title+'日收益率绝对值')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_PATH,"%s.png"%title))
    return

def test():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000021")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['close'].astype(float)
    # hp_ret = df['pctChg'].astype(float)
    ts = hp_ret.dropna()
    ts1= ts.diff(1).dropna()

    ljr = acorr_ljungbox(ts1,lags=20,return_df=True) ## 即白噪音出现的概率
    #如果检验的p值大于0.05 不能拒绝原假设，序列白噪声检验通过
    print(ljr)
    # return 
    ## 画收益曲线
    ts_plot(ts,lags=30,title="Full Close Price")

    if ljr['lb_pvalue'][10] < 0.05:
        #平稳性检测
        print(u'原始序列的ADF检验结果为：', ADF(hp_ret)) #存在单位根的可能性
        #返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
        # 不能拒绝原假设，即原序列存在单位根
        return

    ##定阶模型
    train_results = sm.tsa.arma_order_select_ic(ts1, ic=['aic', 'bic'], max_ar=5, max_ma=5, trend='nc')
    print('----------- AIC  ----------- \n', train_results.aic_min_order) #建立AIC值最小的模型
    print('----------- BIC  ----------- \n', train_results.bic_min_order)

    ## 模型拟合
    model = ARIMA(ts,order=(2,1,5))
    # model = SARIMAX(ts,order=(2,1,5))
    model_fit = model.fit()
    
    print("==== SUMMARY ====\n", model_fit.summary())
    print("=== sigma2 ====\n",model_fit.sigma2)

    ## 模型预测
    a=model_fit.forecast(5)

    print(dir(model_fit))
    print(a)
    with new_png('plot_predict.png'):
    # fig = model.plot_predict(end="2021-06-01")
        # fig = model_fit.plot_predict(5,len(ts)+10,plot_insample=False)
        print("===================================")
        fig = my_plot_predict(model_fit,1,len(ts)+10)
        print("===================================")

    ## fit 后处理
    print(" ---- 置信区 ---- \n", model_fit.conf_int())

    print(model_fit.predict(5,len(ts)+10,None,'levels',False))
    # print(model_fit.predict(pd.to_datetime('2020-07-20'),pd.to_datetime('2021-05-25'),None,'levels',False))

    print(type(model_fit.predict(5,len(ts)+10,None,'levels',False)))

    forecast = model_fit.predict(5,len(ts)+10,None,'levels',False)

    print(ts)
    print(ts.index)
    print(forecast.index)

    # with new_png('plot_predict_fun.png'):
    #     model_fit.predict().plot()

def my_plot_predict(model_fit,start=None, end=None, exog=None, dynamic=False,
                     alpha=.05, plot_insample=True, ax=None,origin_ts=None):
    from statsmodels.graphics.utils import _import_mpl, create_mpl_ax
    _ = _import_mpl()
    fig, ax = create_mpl_ax(ax)

    # use predict so you set dates
    forecast = model_fit.predict(start, end, exog, 'levels', dynamic)
    # doing this twice. just add a plot keyword to predict?
    start, end, out_of_sample, _ = (
        model_fit.model._get_prediction_index(start, end, dynamic))

    if out_of_sample:
        steps = out_of_sample
        fc_error = model_fit._forecast_error(steps)
        conf_int = model_fit._forecast_conf_int(forecast[-steps:], fc_error,
                                            alpha)

    if hasattr(model_fit.data, "predict_dates"):
        from pandas import Series
        forecast = Series(forecast, index=model_fit.data.predict_dates)
        print(model_fit.data.predict_dates)
        ax = forecast.plot(ax=ax, label='forecast')
    else:
        ax.plot(forecast)

    x = ax.get_lines()[-1].get_xdata()
    if out_of_sample:
        label = "{0:.0%} confidence interval".format(1 - alpha)
        ax.fill_between(x[-out_of_sample:], conf_int[:, 0], conf_int[:, 1],
                        color='gray', alpha=.5, label=label)

    if plot_insample:
        import re
        k_diff = model_fit.k_diff
        label = re.sub(r"D\d*\.", "", model_fit.model.endog_names)
        levels = unintegrate(model_fit.model.endog,
                                model_fit.model._first_unintegrate)
        ax.plot(x[:end + 1 - start],
                levels[start + k_diff:end + k_diff + 1], label=label)

    ax.legend(loc='best')

    return fig


def sim_test():
    # 模拟ARCH时间序列
    np.random.seed(2)
    a0 = 2
    a1 = .5
    y = w = np.random.normal(size=1000)
    Y = np.empty_like(y)
    for t in range(1,len(y)):
        Y[t] = w[t] * np.sqrt((a0 + a1*y[t-1]**2))

    ## 画 原始序列的 自相关系数 偏自相关系数 QQ图 PP图
    ts_plot(Y, lags=30,title='模拟ARCH序列')
    # ## 
    # ret_plot(ts, title='沪深300')

    # 模拟GARCH(1, 1) 过程
    np.random.seed(1)
    a0 = 0.2
    a1 = 0.5
    b1 = 0.3
    n = 10000
    w = np.random.normal(size=n)
    garch = np.zeros_like(w)
    sigsq = np.zeros_like(w)
    for i in range(1, n):
        sigsq[i] = a0 + a1*(garch[i-1]**2) + b1*sigsq[i-1]
        garch[i] = w[i] * np.sqrt(sigsq[i])
    print(garch.max())
    ts_plot(garch, lags=30,title='模拟GARCH序列')

    # 使用模拟的数据进行 GARCH(1, 1) 模型拟合
    #arch_model默认建立GARCH（1,1）模型
    am = arch_model(garch)
    res = am.fit(update_freq=0) #update_freq=0表示不输出中间结果，只输出最终结果
    print(res.summary())

    fig = plt.figure(figsize=(10, 8))
    # axes = fig.subplots(1,2)
    layout = (1, 2)
    ax0 = plt.subplot2grid(layout, (0, 0))
    ax1 = plt.subplot2grid(layout, (0, 1))
    # print(res.resid)
    # res.resid.plot(figsize=(12,5),ax=ax0)
    plt.subplot(211)
    plt.plot(res.resid)
    ax0.set_title('拟合GARCH(1,1)残差',size=15)
   
    # res.conditional_volatility.plot(figsize=(12,5),color='r',ax=ax1)
    plt.subplot(212)
    plt.plot(res.conditional_volatility,color='r')
    ax1.set_title('收益率条件方差',size=15)

    plt.savefig(os.path.join(IMAGE_PATH,"模拟GARCH序列 残差 条件方差.png"))


    ## 预测
    forecasts = res.forecast(horizon=5)
    
    plt.figure()
    forecasts.variance.plot() 
    plt.savefig(os.path.join(IMAGE_PATH,"模拟GARCH序列 预测.png"))

    print(res.summary())

    print(forecasts.variance[-1:])
    print(forecasts.mean[-1:])
    print(forecasts.residual_variance[-1:])

def test_reindex():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000021")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['close'].astype(float)
    # hp_ret = df['pctChg'].astype(float)
    ts = hp_ret.dropna()

    model = ARIMA(ts,order=(2,1,5))
    # model = SARIMAX(ts,order=(2,1,5))
    model_fit = model.fit()

    forecast = model_fit.predict(1,len(ts)+0,None,'levels',False)
    # forecast = pd.Series(forecast.values,index=ts.index)
    # print(forecast.reindex(ts.index))
    # with new_png('test_reindex_date.png'):
    #     forecast.plot()
        # ts.plot()

    print(ts)
    print(forecast)
    # print(ts[1:].values - forecast[:-1].values)


    origin_ts = ts[1:].values 
    forecast_ts = forecast[:-1].values
    ts_pchg = ts.diff()/ts * 100    ## DataFrame只要有一个带时间的index 则Df默认使用此时间index 
    print(ts_pchg)

    fpchg = (forecast_ts - ts[0:-1].values) / ts[0:-1].values * 100

    df = pd.DataFrame({'close': origin_ts, 'forecast': forecast_ts,'pchg': ts_pchg[1:],'fpchg': fpchg, 'up': (ts[1:].values - ts[:-1].values) > 0.02 , 'fup': (forecast[:-1].values - ts[:-1].values) > 0.1,'down': (ts[1:].values - ts[:-1].values) < 0} )

    print(df)

    sdf = df[(df['up']==True) & (df['fup']==True) & (df['fpchg']>=1)]
    print(sdf)
    print(len(sdf))

    sdf = df[(df['down']==True) & (df['fup']==True) & (df['fpchg']>=1)]
    print(sdf)
    print(len(sdf))



if __name__ == "__main__":
    # sim_test()
    # test()

    test_reindex()