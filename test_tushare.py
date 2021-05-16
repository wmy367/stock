import os
import tushare

TOKEN_PATH = os.path.join(os.path.split( os.path.realpath(__file__))[0],"./tushare_token")
TUSHARE_TOCKEN = open(TOKEN_PATH).read()
tushare.set_token(TUSHARE_TOCKEN)
fn = 'ts_code ,symbol ,name ,area ,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs'

def test():
    pro = tushare.pro_api()
    data = pro.stock_basic(exchange='', list_status='L', fields=fn)

    print(data)
    data.to_csv('cn_stock.csv')

if __name__=="__main__":
    test()