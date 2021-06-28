

## 使用Python 对 量化金融 R初级教程 的 风险管理波动率预测做迁移

import os,sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
#tsa为Time Series analysis缩写
# import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_lm

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

IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./intro2R")

import contextlib
@contextlib.contextmanager
def new_png(name):
    plt.figure()
    
    yield
    plt.savefig(os.path.join(IMAGE_PATH,name))



def test():
    intc = pd.read_csv(os.path.join(path,"./intro2R/intc.csv"),index_col='date')
    intc.index = pd.to_datetime(intc.index)
    intc = intc.dropna()
    print(intc)
    ## 月收益图
    plt.figure()
    intc.plot()
    plt.savefig(os.path.join(IMAGE_PATH,"月收益"))
    # 在平方收益率的前 12 阶滞后值上执行 Ljung-Box 检验。
    lj =acorr_ljungbox(intc**2,lags=12,return_df=True)
    print(lj)
    # LM 检验
    # lmj = acorr_lm(np.abs(intc), nlags=12)
    # print(lmj)

    # GARCH 模型设定
    am = arch_model(intc)
    res = am.fit(update_freq=0) #update_freq=0表示不输出中间结果，只输出最终结果
    print(res.summary())

    ## 预测
    forecasts = res.forecast(horizon=5)
    
    print(forecasts.variance[-1:])
    print(forecasts.mean[-1:])
    print(forecasts.residual_variance[-1:])

def test_db():
    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000019")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['pctChg'].astype(float)
    ts = hp_ret.dropna()
    
    intc = ts
    print(intc)
    ## 月收益图
    plt.figure()
    intc.plot()
    plt.savefig(os.path.join(IMAGE_PATH,"DB月收益"))
    # 在平方收益率的前 12 阶滞后值上执行 Ljung-Box 检验。
    lj =acorr_ljungbox(intc**2,lags=12,return_df=True)
    print(lj)
    # LM 检验
    # lmj = acorr_lm(np.abs(intc), nlags=12)
    # print(lmj)

    # GARCH 模型设定
    am = arch_model(intc)
    res = am.fit(update_freq=0) #update_freq=0表示不输出中间结果，只输出最终结果
    print(res.summary())

    ## 预测
    forecasts = res.forecast(horizon=5)
    
    print(forecasts.variance[-1:])
    print(forecasts.mean[-1:])
    print(forecasts.residual_variance[-1:])

    print(intc[-5:-1])


if __name__=="__main__":
    # test()

    test_db()