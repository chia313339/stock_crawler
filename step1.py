from bs4 import BeautifulSoup
import pandas as pd
import requests
import configparser
from sqlalchemy import create_engine
from models import *


config = configparser.ConfigParser()
config.read("config.ini")
hostname = config['MYSQL']['hostname']
dbname = config['MYSQL']['dbname']
uname = config['MYSQL']['uname']
pwd = config['MYSQL']['pwd']

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=hostname, db=dbname, user=uname, pw=pwd))



def get_data():
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    res = requests.get('https://histock.tw/stock/rank.aspx?p=all',headers = headers)
    print(res)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

def get_stock_list(soup):
    df = pd.DataFrame()
    for tmp in soup.select('tr')[1::]:
        stock_no = tmp.select('td')[0].text
        stock_name = tmp.select('a')[0].text
        df = df.append(pd.DataFrame({'STOCK_NO':stock_no,'STOCK_NAME':stock_name},index=[0]))
    df = df.reset_index(drop=True)
    return df

def init_write_mysql(engine,df,table_name):
    df.to_sql(table_name, engine, index= False)

def write_mysql(engine,df,table_name):
    df.to_sql(table_name+'_TMP', engine, index= False)
    engine.execute('rename table '+table_name+' to '+table_name+'_OLD')
    engine.execute('rename table '+table_name+'_TMP to '+table_name)
    engine.execute('drop table '+table_name+'_OLD')

def etl_update(table_name):
    engine.execute("UPDATE stock.ETL_INFO SET ETL_DATE=CURRENT_TIMESTAMP() where TABLE_NAME='"+table_name+"'")

def etl_insert(table_name):
    engine.execute("INSERT INTO stock.ETL_INFO (TABLE_NAME, ETL_DATE) VALUES ('"+table_name+"', CURRENT_TIMESTAMP())")


def runapp():
    soup  = get_data()
    df = get_stock_list(soup)
    print("Finish crawler!!!")
    write_mysql(engine,df,'STOCK_LIST')
    print("Finish write table!!!")
    etl_update('STOCK_LIST')
    query = '''select * from stock.ETL_INFO;'''
    df = pd.read_sql(query, con=engine)
    print(df)
    # return 0


if __name__ == '__main__':
    runapp()
