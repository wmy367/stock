from arch import arch_model
import datetime as dt
import pandas_datareader.data as web
import pandas as pd
import os
import matplotlib.pyplot as plt

#正常显示画图时出现的中文和负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
IMAGE_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./images")
# start = dt.datetime(2000,1,1)
# end = dt.datetime(2014,1,1)
# sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
# print(sp500)
# sp500.to_csv("sp500.csv")
# returns = 100 * sp500['Adj Close'].pct_change().dropna()
# am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

def test():
    sp500 = pd.read_csv('sp500.csv',index_col='Date')
    sp500.index = pd.to_datetime(sp500.index)
    print(sp500)
    returns = 100 * sp500['Adj Close'].pct_change().dropna()
    am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

    split_date = dt.datetime(2010,1,1)
    res = am.fit(last_obs=split_date)  
    ## 对波动率和标准化的残差进行绘图
    plt.figure()
    res.plot()
    plt.savefig(os.path.join(IMAGE_PATH,"对波动率和标准化的残差进行绘图.png"))

    ## 预测
    forecasts = res.forecast(horizon=5, start=split_date)
    
    plt.figure()
    forecasts.variance[split_date:].plot() 
    plt.savefig(os.path.join(IMAGE_PATH,"官方arch代码.png"))

    print(res.summary())

    print(forecasts.variance[-1:])
    print(forecasts.mean[-1:])
    print(forecasts.residual_variance[-1:])


if __name__=="__main__":
    test()