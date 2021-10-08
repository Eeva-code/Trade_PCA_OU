import numpy as np
import pandas as pd
import math
from datetime import timedelta, datetime
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA

#------------------------------------------------- Data ------------------------------------------------------#
def get_data():
    path = '/Users/wzhia/Desktop/'
    data = pd.read_csv(path+'data.csv',index_col=0)
    pd.to_datetime(data['date'])

    # Store data into multiple data frames based on ticker name
    uniquetickers = data.index.unique()
    maxlen = max(len(data[data.index==i]) for i in uniquetickers)

    # Create 2 new data frames, one stores last price and the other stores volume of all tickers
    df = pd.DataFrame(None,index=data['date'][:maxlen],columns=uniquetickers) # df stores last price
    df_vol = pd.DataFrame(None,index=data['date'][:maxlen],columns=uniquetickers) # df_vol stores volume
    for i in range(len(uniquetickers)):
        n = data[data.index == uniquetickers[i]].shape[0]
        df[uniquetickers[i]][maxlen-n:] = np.array(data['last'][data.index == uniquetickers[i]])
        df_vol[uniquetickers[i]][maxlen-n:] = np.array(data['volume'][data.index == uniquetickers[i]])

    # take returns
    # dfRet = np.log(((df / df.shift(1)).dropna(how='all')).astype(float))
    return df, df_vol

def trade_data(trading_start, window=60):
    data, volume = get_data()

    trading_start_dt = datetime.strptime(trading_start, '%Y-%m-%d')
    trading_start_window_dt = trading_start_dt - timedelta(window) # trading_start-window
    trading_start_window = datetime.strftime(trading_start_window_dt, '%Y-%m-%d')
    trading_end_dt = trading_start_dt + timedelta(365)
    trading_end = datetime.strftime(trading_end_dt, '%Y-%m-%d')

    estimation_start_dt = trading_start_dt - timedelta(365)
    estimation_start = datetime.strftime(estimation_start_dt, '%Y-%m-%d')

    df_estimation = data.loc[estimation_start:trading_start]
    df_trade = data.loc[trading_start_window:trading_end]

    return df_estimation, df_trade, volume

#---------------------------------------------- PCA -------------------------------------------------------#
def pca(X,w,reqExp):
    X_ori = X
    X = np.array(X)
    X = X[:math.floor(len(X) * w)] # use half of the data for PCA analysis (w = 0.5)
    cov_mat = np.cov(X.T) # Compute covariance matrix
    eigval, eigvec = np.linalg.eig(cov_mat) # Compute eigenvalues and eigenvectors
    n = np.where((np.cumsum(eigval)/sum(eigval))>=reqExp,0,1).sum()+1 # Get min number of eigenvectors to cover exp power
    print("Minimum number of eigenvectors to cover the required explanatory power: ",n)
    pc = eigvec.T[:n] # principal components
    sigma = X.std() # compute volatility (standard deviation)
    pc_adj = pc / np.array(sigma) # Adjust principal components by volatility
    components = pd.DataFrame(np.dot(pc_adj, X_ori.T), columns=X_ori.index)
    return n, components

def standardise_return(Ret,pc):
    Ret = np.array(Ret)
    pc = np.array(pc)
    pc_std = []
    Ret_std = []
    for i in range(pc.shape[0]):
        pc_std.append((pc[i] - pc[i].mean()) / (pc[i].std()))  # standardize returns
    for i in range(Ret.shape[1]):
        Ret_std.append((Ret[:, i] - Ret[:, i].mean()) / (Ret[:, i].std()))  # standardize returns
    return Ret_std, pc_std
#-------------------------------------------- NOT COMPLETE Strategy ----------------------------------------------------#
# 1. OU stochastic process: fit the residual of each stock
#     a. Multi-linear regression on the stock returns and the principal components to obtain betas and residuals.
#     b. Get parameters of OU-estimation on each stock, the parameters measures the mean-reversion speed.
#     c. Analysis: Use R-square to measure robustness
# 2. Portfolio selection:
#     a. Get the stocks that have the fastest mean-conversion rate, and should be traded in the next level.
#     b. Get positions of each stocks in the whole trading periods.
#         - Compute trading signal of each stock
#         - Allocate funds on each stock that has a trading signal, whether long or short
def fit_ols(y,X):
    beta = []
    Rsquare = []
    Residuals = []
    X = pd.DataFrame(X)
    X = sm.add_constant(X.T)
    for i in range(len(y)):
        model = sm.OLS(y[i], X)
        results = model.fit()
        beta.append(results.params)
        Rsquare.append(results.rsquared)
        Residuals.append(results.resid)
    return beta, Rsquare, residuals

def ou_estimation(residual):
    SST = residual.var()
    R_square, k = [], []

    for i in range(residual.shape[1]):
        arma = ARMA(residual.iloc[:, i], order=(1, 0)).fit(maxiter=100, disp=-1)
    a = arma.params[0]
    b = arma.params[1]

    #xx
    return Rsquare



def portfolio_selection():

    return portfolio_index

def trading_signal():
    return ??


def earnings():
    return

def get_position():
    return
#------------------------------------------- Backtesting --------------------------------------------------#




#----------------------------------------------- Main -----------------------------------------------------#
def trade():
    dfRet, df_vol = get_data() # last price and volume of all tickers
    Ret = dfRet[dfRet.columns[~dfRet.isnull().any()]] # get tickers with same date length (without NAs)
    n, pc = pca(Ret,w=0.5,reqExp=0.7)  # PCA analysis on the returns, default explanatory power is 0.7, w denotes the fraction of stocks used to do PCA
    Ret_std, pc_std = standardise_return(Ret,pc)
    beta, Rsq, residual = fit_ols(Ret_std,pc_std)
    a, b = get_residual(Ret_std,pc_std)
    portfolio_index = portfolio_selection(pc)
    position = get_position(portfolio_index)
    cum_pnl, sharp_total = earnings(dfRet, position)
    return sharp_total

if __name__ == '__main__':
    time_list = ['2015-01-04', '2016-01-04', '2017-01-04', '2018-01-04', '2019-01-04',
                 '2019-01-04', '2020-01-04','2021-01-04']
    sharp_all = []
    for i in time_list:
        sharp = trade(i)
        sharp_all.append(sharp)
    print(sharp_all)

