
import os,sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
#tsa为Time Series analysis缩写
# import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
from statsmodels.stats.diagnostic import acorr_ljungbox

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

IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./images")

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

def whitenoise_test(ts):
    '''计算box pierce 和 box ljung统计量'''
    q,p=acorr_ljungbox(ts,return_df=False)
    with plt.style.context('ggplot'):    
        fig = plt.figure(figsize=(10, 4))
        axes = fig.subplots(1,2)
        axes[0].plot(q, label='Q统计量')
        axes[0].set_ylabel('Q')
        axes[1].plot(p, label='p值')
        axes[1].set_ylabel('P')
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_PATH,"whitenoise_test.png"))
    return


def test():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000019")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    # hp_ret = df['close'].astype(float)
    hp_ret=np.log(df.close/df.close.shift(1))
    ts = hp_ret.dropna() 

    ts_plot(ts,lags=30,title="收益")


    # 模拟ARCH时间序列
    np.random.seed(2)
    a0 = 2
    a1 = .5
    y = w = np.random.normal(size=1000)
    Y = np.empty_like(y)
    for t in range(1,len(y)):
        Y[t] = w[t] * np.sqrt((a0 + a1*y[t-1]**2))
    ts_plot(Y, lags=30,title='模拟ARCH')

    ret_plot(ts, title='沪深300')

    ret = ts
    whitenoise_test(ret**2)

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
    ts_plot(garch, lags=30,title='模拟GARCH')

    # 使用模拟的数据进行 GARCH(1, 1) 模型拟合
    #arch_model默认建立GARCH（1,1）模型
    am = arch_model(garch)
    res = am.fit(update_freq=0)

    print(res.summary())

    #拟合沪深300收益率数据

    Y=ret*100.0
    am = arch_model(Y,p=1, o=1, q=1, dist='StudentsT')
    res = am.fit(update_freq=0)
    #update_freq=0表示不输出中间结果，只输出最终结果
    print(res.summary())

    fig = plt.figure(figsize=(10, 8))
    # axes = fig.subplots(1,2)
    layout = (1, 2)
    ax0 = plt.subplot2grid(layout, (0, 0))
    ax1 = plt.subplot2grid(layout, (0, 1))
    res.resid.plot(figsize=(12,5),ax=ax0)
    ax0.set_title('沪深300收益率拟合GARCH(1,1)残差',size=15)
   
    res.conditional_volatility.plot(figsize=(12,5),color='r',ax=ax1)
    ax1.set_title('沪深300收益率条件方差',size=15)

    plt.savefig(os.path.join(IMAGE_PATH,"GARCH.png"))

    # 基于估计模型将预测结果图形化
    plt.figure()
    res.hedgehog_plot(type='volatility',horizon=3)
    plt.savefig(os.path.join(IMAGE_PATH,"基于估计模型将预测结果图形化.png"))

    ## 预测VaR
    res_forecast = res.forecast(horizon=5)
    print(res_forecast.mean[-1:])
    print(res_forecast.variance[-1:])
    print(res_forecast.residual_variance[-1:])





if __name__=="__main__":
    test()
