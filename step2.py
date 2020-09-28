from bs4 import BeautifulSoup
import pandas as pd
import requests
import configparser
from sqlalchemy import create_engine
from models import *
import tqdm
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
import datetime
import yfinance as yf


config = configparser.ConfigParser()
config.read("config.ini")
hostname = config['MYSQL']['hostname']
dbname = config['MYSQL']['dbname']
uname = config['MYSQL']['uname']
pwd = config['MYSQL']['pwd']

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=hostname, db=dbname, user=uname, pw=pwd))



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


def detailed_linear_regression(X, y):
    """
    Assume X is array-like with shape (num_samples, num_features)
    Assume y is array-like with shape (num_samples, num_targets)
    Computes the least-squares regression model and returns a dictionary consisting of
    the fitted linear regression object; a series with the residual standard error,
    R^2 value, and the overall F-statistic with corresponding p-value; and a dataframe
    with columns for the parameters, and their corresponding standard errors,
    t-statistics, and p-values.
    """
    # Create a linear regression object and fit it using x and y
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Store the parameters (regression intercept and coefficients) and predictions
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(X)
    
    # Create matrix with shape (num_samples, num_features + 1)
    # Where the first column is all 1s and then there is one column for the values
    # of each feature/predictor
    X_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    
    # Compute residual sum of squares
    RSS = np.sum((y - predictions)**2)
    
    # Compute total sum of squares
    TSS = np.sum((np.mean(y) - y)**2)
    
    # Estimate the variance of the y-values
    obs_var = RSS/(X_mat.shape[0] - X_mat.shape[1])
    
    # Residual standard error is square root of variance of y-values
    RSE = obs_var**0.5
    
    # Variances of the parameter estimates are on the diagonal of the 
    # variance-covariance matrix of the parameter estimates
    var_beta = obs_var*(np.linalg.inv(np.matmul(X_mat.T, X_mat)).diagonal())
    
    # Standard error is square root of variance
    se_beta = np.sqrt(var_beta)
    
    # t-statistic for beta_i is beta_i/se_i, 
    # where se_i is the standard error for beta_i
    t_stats_beta = params/se_beta
    
    # Compute p-values for each parameter using a t-distribution with
    # (num_samples - 1) degrees of freedom
    beta_p_values = [2 * (1 - stats.t.cdf(np.abs(t_i), X_mat.shape[0] - 1))
                    for t_i in t_stats_beta]
    
    # Compute value of overall F-statistic, to measure how likely our
    # coefficient estimate are, assuming there is no relationship between
    # the predictors and the response
    F_overall = ((TSS - RSS)/(X_mat.shape[1] - 1))/(RSS/(X_mat.shape[0] - X_mat.shape[1]))
    F_p_value = stats.f.sf(F_overall, X_mat.shape[1] - 1, X_mat.shape[0] - X_mat.shape[1])
    
    # Construct dataframe for the overall model statistics:
    # RSE, R^2, F-statistic, p-value for F-statistic
    oa_model_stats = pd.Series({"Residual standard error": RSE, "R-squared": reg.score(X, y),
                                "F-statistic": F_overall, "F-test p-value": F_p_value})
    
    # Construct dataframe for parameter statistics:
    # coefficients, standard errors, t-statistic, p-values for t-statistics
    param_stats = pd.DataFrame({"Coefficient": params, "Standard Error": se_beta,
                                "t-value": t_stats_beta, "Prob(>|t|)": beta_p_values})
    return {"model": reg, "param_stats": param_stats, "oa_stats": oa_model_stats}



def get_stock_history(stock_no):
    DT_DIFF = 365.25*3.5
    END_DT = datetime.datetime.now()
    START_DT = END_DT+datetime.timedelta(days=-DT_DIFF)
    STOCK_TW = stock_no+'.TW'
    df = yf.Ticker(STOCK_TW).history(start=START_DT.strftime("%Y-%m-%d"), end=END_DT.strftime("%Y-%m-%d"))
#     print('Get stock history of '+STOCK_TW+' finish!!')
    return df


def get_fiveline_table(stock_history,prob1 = 68.26,prob2 = 95.44):
    s1 = (1-prob2/100)/2
    s2 = (1-prob1/100)/2
    s3 = 1-s2
    s4 = 1-s1
    x = (np.arange(len(stock_history))+1).reshape((-1, 1))
    y = np.array(stock_history.Close)
    stock_reg = detailed_linear_regression(x,y)
    reg_line = stock_reg['model'].predict(x)
    ss1 = norm.ppf(s1,reg_line,stock_reg['oa_stats'][0])
    ss2 = norm.ppf(s2,reg_line,stock_reg['oa_stats'][0])
    ss3 = norm.ppf(s3,reg_line,stock_reg['oa_stats'][0])
    ss4 = norm.ppf(s4,reg_line,stock_reg['oa_stats'][0])
    Rsquared = stock_reg['oa_stats'][1]
    Slope = stock_reg['model'].coef_[0]
    Sd = stock_reg['oa_stats'][0]
    
    fiveline_data = pd.DataFrame({'no' : np.arange(len(x))+1,
                              'stock_date':stock_history.index,
                              'stock_price':stock_history.Close,
                              'ss1':ss1,
                              'ss2':ss2,
                              'reg_line':reg_line,
                              'ss3':ss3,
                              'ss4':ss4,
                              'Rsquared':Rsquared,
                              'Slope':Slope,
                              'Sd':Sd
                              })
    return fiveline_data


def assess_status(fiveline_vec):
    intervals = np.array(fiveline_vec)[0][3:8]
    val = intervals > fiveline_vec.stock_price[0]
    if(sum(val)==0): status = "股價高於歷史股價2倍標準差之外，絕對高點，建議看空。"
    if(sum(val)==1): status = "股價高於歷史股價1倍標準差之外，相對高點，建議持續觀察或看空。"
    if(sum(val)==2): status = "股價位於一般波動區間，且相對高點，建議持續觀察。"
    if(sum(val)==3): status = "股價位於一般波動區間，且相對低點，建議持續觀察。"
    if(sum(val)==4): status = "股價低於歷史股價1倍標準差之外，相對低點，建議持續觀察或看多。"
    if(sum(val)==5): status = "股價低於歷史股價2倍標準差之外，絕對低點，建議看多。"
    return status,sum(val)

def get_stock_status(stock_name):
    try:
        stock_his = get_stock_history(stock_name)
        stock_fivetb = get_fiveline_table(stock_his)
        status = assess_status(stock_fivetb.tail(1))
        tmp = stock_fivetb.tail(1).drop(['no'], axis=1)
        tmp['status'] = status[0]
        tmp['status_cd'] = status[1]
        tmp['stock_name'] = stock_name
        return tmp
    except:
        next
#         print("===== Stock: "+stock_name+" Error!!! =====")

def get_all_stock_status():
    stock_list = pd.read_sql("select * from stock.STOCK_LIST", engine).STOCK_NO
    stock_all_status = pd.DataFrame()
    for st_code in stock_list:
        tmp = get_stock_status(st_code)
        stock_all_status = stock_all_status.append(tmp)
    stock_all_status = stock_all_status.reset_index(drop=True)
    return stock_all_status


def runapp():
    df_stock_status = get_all_stock_status()
    print("Finish all stock status access!!!")
    write_mysql(engine,df_stock_status,'STOCK_STATUS')
    print("Finish write table!!!")
    etl_update('STOCK_STATUS')
    query = '''select * from stock.ETL_INFO;'''
    df = pd.read_sql(query, con=engine)
    print(df)
    # return 0


if __name__ == '__main__':
    runapp()
