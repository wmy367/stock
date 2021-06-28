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
apath = os.path.join(path,'../data_api')
if apath not in sys.path: sys.path.append(apath)

import api_baostock

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

def analysisStock(code="sz.000001",start_date=None, end_date=None, key='close'):
    df = api_baostock.baoStockDataFrame(code=code,start_date=start_date, end_date=end_date)
    df.index = pd.to_datetime(df.date)

    hp_ret = df[key].astype(float)
    ts = hp_ret.dropna()
    ts1= ts.diff(1).dropna()

    if len(ts) < 100:
        return False 

    ljr = acorr_ljungbox(ts1,lags=24,return_df=True) ## 即白噪音出现的概率
    #如果检验的p值大于0.05 不能拒绝原假设，序列白噪声检验通过
    if not np.min(ljr['lb_pvalue'][0:12]) > 0.05:
        print("Code: {} acorr_ljungbox False".format(code))
        print(ljr)
        return False

    ##定阶模型
    train_results = sm.tsa.arma_order_select_ic(ts1, ic=['aic', 'bic'], max_ar=5, max_ma=5, trend='nc')
    print('----------- AIC  ----------- \n', train_results.aic_min_order) #建立AIC值最小的模型
    print('----------- BIC  ----------- \n', train_results.bic_min_order)

    ## 模型拟合
    model = ARIMA(ts,order=(train_results.aic_min_order[0],1,train_results.aic_min_order[1]))
    # model = SARIMAX(ts,order=(2,1,5))
    try:
        model_fit = model.fit(update_freq=0)
    except Exception as e:
        return False

    print("==== SUMMARY ====\n", model_fit.summary())
    print("=== sigma2 ====\n",model_fit.sigma2)

    ## 模型预测
    a=model_fit.forecast(5)
    with new_png('plot_predict_%s.png'%(code,)):
        fig = model_fit.plot_predict(50,len(ts)+4)


    ## 波动率建模
    Y = ts1
    am = arch_model(Y,p=1, o=1, q=1, dist='StudentsT')
    res = am.fit(update_freq=0)
    #update_freq=0表示不输出中间结果，只输出最终结果
    print(res.summary())

    ## 价值判断
    return ret_values(model_fit,ts)


## 判断 回报盈利 是否值得
def ret_values(model_fit,origin_ts):
    ts = origin_ts
    ## 预测一天，输出序列长度 为 len(origin_ts)
    forecast = model_fit.predict(1,len(ts)+0,None,'levels',False)

    ## 使origin 和 forecast 长度一致，origin去掉第一个，forecast去掉最后一个
    origin_ts = ts[1:].values 
    forecast_ts = forecast[:-1].values
    ts_pchg = ts.diff()/ts * 100    ## DataFrame只要有一个带时间的index 则Df默认使用此时间index 

    fpchg = (forecast_ts - ts[0:-1].values) / ts[0:-1].values * 100 # 预测的盈利率
    ## 
    # up 是否涨
    # down 是否跌
    # fup 预测涨
    df = pd.DataFrame({'origin': origin_ts, 'forecast': forecast_ts,'pchg': ts_pchg[1:],'fpchg': fpchg, 'up': (ts[1:].values - ts[:-1].values) > 0.02 , 'fup': (forecast[:-1].values - ts[:-1].values) > 0.1,'down': (ts[1:].values - ts[:-1].values) < 0} )

    ## 预测和实际匹配 盈利
    mdf = df[(df['up']==True) & (df['fup']==True) & (df['fpchg']>=1)]
    ## 预测与实际相反 亏损
    ndf = df[(df['pchg'] < -0.3 ) & (df['fup']==True) & (df['fpchg']>=1)]

    return {
        'buyPoints' : len(df[ df['fpchg'] >= 1 ]) / len(df) * 100,
        'matchUp' : len(mdf) / (1+len(df[ df['fpchg'] >= 1 ])),
        'noMathUp': len(ndf) / (1+len(df[ df['fpchg'] >= 1 ])),
    }
    


def test():
    print(analysisStock())


if __name__ == "__main__":
    test()
