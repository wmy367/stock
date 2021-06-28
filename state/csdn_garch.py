import os 
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.graphics.tsaplots import plot_pacf    #
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from arch.univariate import arch_model

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
    plt.rcParams['font.sans-serif'] = ['SimHei'] #字体为黑体
    plt.rcParams['axes.unicode_minus'] = False #正常显示负号 #时序图的绘制
    
    yield
    plt.savefig(os.path.join(IMAGE_PATH,name))

def test():
    plt.rcParams['font.sans-serif'] = ['SimHei'] #字体为黑体
    plt.rcParams['axes.unicode_minus'] = False #正常显示负号 #时序图的绘制

    eg,_ = db.CnnDatabase('baoStock')
    df = test_baostock.dayTableFromeDataBase(db_engin=eg, code="sz.000019")
    df = df.sort_values(by='date', ascending=False)[0:200]
    df = df.sort_values(by='date', ascending=True)
    df.index = pd.to_datetime(df.date)
    # print(df)
    ## 计算简单回报率
    # hp_ret = df['pctChg']
    hp_ret = df['close'].astype(float)
    ts = hp_ret 
    lbvalue,pvalue = acorr_ljungbox(ts,lags=30) ## 即白噪音出现的概率
    #如果检验的p值大于0.05 不能拒绝原假设，序列白噪声检验通过

    ## 
    with new_png('origin.png'):
        ts.plot()

    ts1 = ts.diff().dropna()
    with new_png('acf.png'):
        plot_acf(ts1)
        plot_pacf(ts1)

    print(lbvalue,pvalue)

    ##定阶模型
    train_results = sm.tsa.arma_order_select_ic(ts1, ic=['aic', 'bic'], max_ar=4, max_ma=4, trend='nc')
    print('AIC', train_results.aic_min_order) #建立AIC值最小的模型
    print('BIC', train_results.bic_min_order)

    ## 模型拟合
    model = ARIMA(ts,order=(3,1,2))
    model_fit = model.fit()
    print("==== SUMMARY ====\n", model_fit.summary())
    print("=== sigma2 ====\n",model_fit.sigma2)

    ## 残差序列检验
    stdresid=model_fit.resid/math.sqrt(model_fit.sigma2) #标准化残差
    xstdresid=model_fit.resid # 残差
    with new_png('标准化残差序列图.png'):
        plt.plot(stdresid)
        plt.xticks(rotation=45) #坐标角度旋转
        plt.xlabel('日期') #横、纵坐标以及标题命名
        plt.ylabel('标准化残差')
        plt.title('标准化残差序列图',loc='center')

    with new_png('残差序列图.png'):
        plt.plot(xstdresid)
        plt.xticks(rotation=45) #坐标角度旋转
        plt.xlabel('日期') #横、纵坐标以及标题命名
        plt.ylabel('残差')
        plt.title('残差序列图',loc='center')

    with new_png('残差序列图平方.png'):
        plt.plot(xstdresid**2)
        plt.xticks(rotation=45) #坐标角度旋转
        plt.xlabel('日期') #横、纵坐标以及标题命名
        plt.ylabel('残差**2')
        plt.title('残差**2序列图',loc='center')

    ## 构建 Garch(1,1) 模型
    garch = arch_model(model_fit.resid,mean="Constant",p=1,o=0,q=1,vol="Garch")
    garch_fit = garch.fit()

    print("==== GARCH =====\n",garch_fit)

if __name__=="__main__":
    test()