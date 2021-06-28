import os,sys
import re
import datetime
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
apath = os.path.join(path,'../state')
if apath not in sys.path: sys.path.append(apath)

import stock_arima

CN_STOCK = os.path.join(os.path.split( os.path.realpath(__file__))[0],"../cn_stock.csv")

def all_stock_als():
    end_date = datetime.date.today().__format__('%Y-%m-%d')
    df = pd.read_csv(CN_STOCK, index_col=0)
    start = 0
    allDF = {}
    allDF['name'] = []
    allDF['code'] = []
    allDF['buyPoints'] = []
    allDF['matchUp'] = []
    allDF['noMathUp'] = []

    import warnings
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                            FutureWarning)
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                            FutureWarning)

    for index in range(start,len(df['ts_code'])):
        n = df['ts_code'][index]
        m0 = re.match(r'(\w+)\.(\w+)', n)
        info = 'ignore'

        if m0:
            if m0.group(2).lower() == "sz":
                precode = "1"
            else:
                precode = "0"
            
            code = precode + m0.group(1)
            info = 'done'

            code="%s.%s"%(m0.group(2).lower(),m0.group(1))
            print("====================== [%d]"%index,code)
            rel = stock_arima.analysisStock(code=code,start_date=None, end_date=end_date, key='close')
            if rel:
                print("--- code:{} ----".format(code))
                print(rel)
                allDF['name'].append(df['name'][index])
                allDF['code'].append(code)
                allDF['buyPoints'].append(rel['buyPoints'])
                allDF['matchUp'].append(rel['matchUp'])
                allDF['noMathUp'].append(rel['noMathUp'])
            else:
                allDF['name'].append(df['name'][index])
                allDF['code'].append(np.nan)
                allDF['buyPoints'].append(np.nan)
                allDF['matchUp'].append(np.nan)
                allDF['noMathUp'].append(np.nan)


    return pd.DataFrame(allDF)

def test0():
    df = all_stock_als()
    df.to_csv("all_stock_analysis.csv")

if __name__ == "__main__":
    test0()



