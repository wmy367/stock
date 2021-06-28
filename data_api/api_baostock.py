import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String,Float,Date,Boolean,Text

import baostock as bs
import pandas as pd
from sqlalchemy import Column, Integer, String,Float,Date,Boolean
import datetime
import numpy
import os
import re
CN_STOCK = os.path.join(os.path.split( os.path.realpath(__file__))[0],"../cn_stock.csv")

Base = declarative_base()

def CnnDatabase(name):
    Base = declarative_base()
    # print(R'sqlite:///'+os.path.join(os.path.split( os.path.realpath(__file__))[0],"./database/%s.sqlite3"%name))
    xENGINE = create_engine(R'sqlite:///'+os.path.join(os.path.split( os.path.realpath(__file__))[0],"../database/%s.sqlite3"%name))
    # ENGINE = engine = create_engine('sqlite:///foo.db')

    Base.metadata.create_all(xENGINE, checkfirst=True)

    xxSession = sessionmaker(bind=xENGINE)

    # 创建Session类实例
    zSession = xxSession()

    return xENGINE,zSession

class StockDayDB(Base):
    # 指定本类映射到users表
    __tablename__ = 'baostock_day'
    # 如果有多个类指向同一张表，那么在后边的类需要把extend_existing设为True，表示在已有列基础上进行扩展
    # 或者换句话说，sqlalchemy允许类是表的字集
    # __table_args__ = {'extend_existing': True}
    # 如果表在同一个数据库服务（datebase）的不同数据库中（schema），可使用schema参数进一步指定数据库
    # __table_args__ = {'schema': 'test_database'}
    
    # 各变量名一定要与表的各字段名一样，因为相同的名字是他们之间的唯一关联关系
    # 从语法上说，各变量类型和表的类型可以不完全一致，如表字段是String(64)，但我就定义成String(32)
    # 但为了避免造成不必要的错误，变量的类型和其对应的表的字段的类型还是要相一致
    # sqlalchemy强制要求必须要有主键字段不然会报错，如果要映射一张已存在且没有主键的表，那么可行的做法是将所有字段都设为primary_key=True
    # 不要看随便将一个非主键字段设为primary_key，然后似乎就没报错就能使用了，sqlalchemy在接收到查询结果后还会自己根据主键进行一次去重
    # 指定id映射到id字段; id字段为整型，为主键，自动增长（其实整型主键默认就自动增长）
    # id = Column(Integer, primary_key=True, autoincrement=True)
    # 指定name映射到name字段; name字段为字符串类形，
    # name = Column(String(20))
    # fullname = Column(String(32))
    # password = Column(String(32))

    vid = Column(String(64), primary_key=True)
    date = Column((Date))
    code = Column(String(32))
    locals()['open'] = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    preclose = Column(Float)
    volume = Column(Integer)
    amount = Column(Float)
    adjustflag = Column(String(16))
    turn = Column(Float)
    tradestatus= Column(String(16))
    pctChg = Column(Float)
    isSt = Column(Boolean)


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
    rs = bs.query_history_k_data_plus(code,
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

def dayTableFromeDataBase(db_engin=None,code="sh.600000",start_date='2021-01-01', end_date='2021-05-15'):
    result = db_engin.execute("select * from sqlite_master where name = 'baostock_day'")
    if len(list(result)) == 0:
        return list(result)
    df = pd.read_sql("SELECT * FROM baostock_day WHERE code = '%s' AND date >= '%s' AND date <= '%s';"%(code,start_date,end_date), db_engin, index_col=None, coerce_float=False, params=None, parse_dates=None, columns=None, chunksize=None)
    # print(df)
    return df

## 默认以当前日期为截止日期获取两年内的数据，并缓存到数据库
def baoStockDataFrame(db_engin=None,code="sh.600000",start_date=None, end_date=None):
    if not end_date:
        end_date = datetime.date.today().__format__('%Y-%m-%d')

    if not start_date:
        start_date = (datetime.datetime.strptime(end_date,'%Y-%m-%d') - datetime.timedelta(days=2*365)).strftime("%Y-%m-%d")

    eg,_ = db_engin or CnnDatabase('baoStock_%s'%end_date)
    ## 先从数据库查询
    df = dayTableFromeDataBase(eg,code=code,start_date=start_date, end_date=end_date)
    if len(df) > 0:
        return df 

    ## 没有则调用网络接口
    dayStockToDateBase(db_engine=eg,clg=None,code=code,start_date=start_date, end_date=end_date,adjustflag="3")

    return dayTableFromeDataBase(eg,code=code,start_date=start_date, end_date=end_date)



def test0():
    print(baoStockDataFrame())



if __name__=="__main__":
    # test3()
    test0()
