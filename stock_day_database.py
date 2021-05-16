import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String,Float,Date,Boolean,Text

Base = declarative_base()
print(R'sqlite:///'+os.path.join(os.path.split( os.path.realpath(__file__))[0],"./database/day_db.sqlite3"))
ENGINE = create_engine(R'sqlite:///'+os.path.join(os.path.split( os.path.realpath(__file__))[0],"./database/day_db.sqlite3"))
# ENGINE = engine = create_engine('sqlite:///foo.db')

Base.metadata.create_all(ENGINE, checkfirst=True)

xSession = sessionmaker(bind=ENGINE)

# 创建Session类实例
Session = xSession()

def CnnDatabase(name):
    Base = declarative_base()
    # print(R'sqlite:///'+os.path.join(os.path.split( os.path.realpath(__file__))[0],"./database/%s.sqlite3"%name))
    xENGINE = create_engine(R'sqlite:///'+os.path.join(os.path.split( os.path.realpath(__file__))[0],"./database/%s.sqlite3"%name))
    # ENGINE = engine = create_engine('sqlite:///foo.db')

    Base.metadata.create_all(xENGINE, checkfirst=True)

    xxSession = sessionmaker(bind=xENGINE)

    # 创建Session类实例
    zSession = xxSession()

    return xENGINE,zSession


class StockDayDB(Base):
    # 指定本类映射到users表
    __tablename__ = 'stock_day'
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    # 指定name映射到name字段; name字段为字符串类形，
    # name = Column(String(20))
    # fullname = Column(String(32))
    # password = Column(String(32))

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

def test0():
    our_user = Session.query(StockDayDB).first()

    print(our_user)

if __name__=="__main__":
    test0()