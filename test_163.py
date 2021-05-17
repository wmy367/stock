import pandas as pd
import os 
from sqlalchemy import Column, Integer, String,Float,Date,Boolean
import datetime
import stock_day_database as db
import numpy
import re
from retry import retry

CN_STOCK = os.path.join(os.path.split( os.path.realpath(__file__))[0],"cn_stock.csv")

 
#沪市前面加0，深市前面加1，比如0000001，是上证指数，1000001是中国平安
@retry(delay=4, jitter=1,tries=100)
def get_daily(code,start='19900101',end=''):
    url_mod="http://quotes.money.163.com/service/chddata.html?code=%s&start=%s&end=%s"
    url=url_mod%(code,start,end)
    df=pd.read_csv(url, encoding = 'gbk')
    return df
 
def test0():
    print(get_daily(code="1000001"))

def test1():
    df = pd.read_csv(CN_STOCK, index_col=0)
    # print(df['ts_code'])
    # index = 0
    start = 0
    # for n in df['ts_code']:
    for index in range(start,len(df['ts_code'])):
        n = df['ts_code'][index]
        # print(n)
        m0 = re.match(r'(\w+)\.(\w+)', n)
        info = 'ignore'
        if m0:
            if m0.group(2).lower() == "sz":
                precode = "1"
            else:
                precode = "0"
            
            code = precode + m0.group(1)
            info = 'done'

            dailyToDatabase(code=code,start='19900101',end='')
        # index += 1
        print("%s %s !!! %d/%d"%(n,info,index+1,len(df['ts_code'])))

def dailyToDatabase(code="1000001",start='19900101',end=''):
    df = get_daily(code,start=start,end=end)
    df.to_csv('tmp.csv')
    # df = pd.read_csv('tmp.csv', index_col=0)
    keys = {
     '日期'   : 'date',  
     '股票代码'    : 'code',
     '名称'    : 'name',
     '收盘价'    : 'close',
     '最高价'    : 'high',
     '最低价'    : 'low',
     '开盘价'    : 'open',
     '前收盘'    : 'preclose',
     '涨跌额'      : 'pctChV',
     '涨跌幅'     : 'pctChg',
     '换手率'        : 'turn',
     '成交量'          : 'volume',
     '成交金额'           : 'amount',
     '总市值'          : 'total',
     '流通市值'    : 'shareTotal',
     '成交笔数' :'anum'
    }
    ndf = {}
    for (k,v) in keys.items():
        # ndf[v] = []
        # df[df[k]=='None'][k] = 0
        for i,x in df[df[k] == 'None'].iterrows():
            df.at[i,k] = numpy.nan

        ndf[v] = df[k]

        # df[['c3','c5']] = df[['c3','c5']].apply(pd.to_numeric)
    # df = df[df['成交笔数'] == 'None'].apply( lambda x : 0)
    # df[df['成交笔数'] == 'None']['成交笔数'] = 0

    
    

    # print(df[df['成交笔数'] != 'None'])

    ndf = pd.DataFrame(ndf)
    ndf[['close','high','low','open','preclose','pctChV','pctChg','turn','amount','total','shareTotal']] = ndf[['close','high','low','open','preclose','pctChV','pctChg','turn','amount','total','shareTotal']].astype(float)

    dtype = {
        'date': Date,
        'code': (String(32)),
        'open':  (Float),
        'high': (Float),
        'low' : (Float),
        'close' : (Float),
        'preclose' : (Float),
        'volume' : (Integer),
        'amount' : (Float),

        'turn' : (Float),

        'pctChg' : (Float),
        'pctChV' : Float,
        'total': Float,
        'shareTotal': Float,
        'anum': Integer()
    }
    ndf['date'] = [ datetime.datetime.strptime(x,'%Y-%m-%d') for x in ndf['date']]
    ndf.to_sql("stock_163_daily", db.ENGINE, schema=None, if_exists='append', index=False, index_label=None, chunksize=None, dtype=dtype, method=None)

    


if __name__=="__main__":
    test1()
    # dailyToDatabase()
