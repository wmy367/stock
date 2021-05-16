import baostock as bs
import pandas as pd
import stock_day_database as db
from sqlalchemy import Column, Integer, String,Float,Date,Boolean
import datetime
import numpy
import os
import re
CN_STOCK = os.path.join(os.path.split( os.path.realpath(__file__))[0],"cn_stock.csv")


def test0():
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 登出系统 ####
    bs.logout()

def test1():
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus("sh.600000",
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date='2021-01-01', end_date='2021-05-15',
        frequency="d", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####   
    # result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    print(result)
    # result[0]

    #### 登出系统 ####
    bs.logout()

def dayStockToDateBase(db_engine=None,clg=None,code="sh.600000",start_date='2021-01-01', end_date='2021-05-15',adjustflag="3"):
    #### 登陆系统 ####
    lg = clg or bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus("sh.600000",
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####   
    # result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    # print(result[1])
    # result[0]
    result['vid'] = result['date'] + code
    result['date'] = [ datetime.datetime.strptime(x,'%Y-%m-%d') for x in result['date']]
    # result[['open','high','low','close','preclose','amount','turn','pctChg']] = result[['open','high','low','close','preclose','amount','turn','pctChg']].astype(float)
    dtype = {
        'vid': String(32),
        'date': Date,
        'code': (String(32)),
        'open':  (Float),
        'high': (Float),
        'low' : (Float),
        'close' : (Float),
        'preclose' : (Float),
        'volume' : (Integer),
        'amount' : (Float),
        'adjustflag' : (String(16)),
        'turn' : (Float),
        'tradestatus': (String(16)),
        'pctChg' : (Float),
        'isST' : (Integer),
    }
    # print(result)
    ndf = pd.DataFrame(result)
    for (k,v) in dtype.items():
        for i,x in result[result[k] == ''].iterrows():
            result.at[i,k] = numpy.nan

        ndf[k] = result[k]

    # ndf = pd.DataFrame(ndf)
    # print(ndf)

    ndf.to_sql("baostock_day", db_engine, schema=None, if_exists='append', index=False, index_label=None, chunksize=None, dtype=dtype, method=None)

    if not clg:
        bs.logout()

def dayTableFromeDataBase(code="sh.60000"):
    df = pd.read_sql("SELECT * FROM stock_day WHERE code = '%s';"%code, db.ENGINE, index_col=None, coerce_float=False, params=None, parse_dates=None, columns=None, chunksize=None)
    print(df)


def test3():
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    eg,_ = db.CnnDatabase('baoStock')
    df = pd.read_csv(CN_STOCK, index_col=0)
    # print(df['ts_code'])
    # index = 0
    start = 0
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

            dayStockToDateBase(db_engine=eg, clg=lg, code="%s.%s"%(m0.group(2),m0.group(1)),start_date="1991-01-01",end_date="")
        # index += 1
        print("%s %s !!! %d/%d"%(n,info,index+1,len(df['ts_code'])))

    dayTableFromeDataBase(code="sh.600000")

    lg.logout()

if __name__=="__main__":
    test3()
