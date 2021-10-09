import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#------------------------------------------------- Data ------------------------------------------------------#
def trade_data():
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

    return df, df_vol

def get_data(trading_start, window=60):
    data, volume = trade_data()

    trading_start_dt = datetime.strptime(trading_start, '%Y-%m-%d')
    trading_end_dt = trading_start_dt + timedelta(365)
    trading_end = datetime.strftime(trading_end_dt, '%Y-%m-%d')

    estimation_start_dt = trading_start_dt - timedelta(365)
    estimation_start = datetime.strftime(estimation_start_dt, '%Y-%m-%d')

    df_all = data.loc[estimation_start:trading_end]
    df_all = df_all[df_all.columns[~df_all.isnull().any()]]

    while not (df_all.index == trading_start).any():
        trading_start_dt += timedelta(1)
        trading_start = datetime.strftime(trading_start_dt, '%Y-%m-%d')

    cutoff = np.where(df_all.index == trading_start)[0][0]
    df_estimation = df_all[:cutoff]
    df_trade = df_all[cutoff-window:]

    return df_all, df_estimation, df_trade, volume
#---------------------------------------------- PCA and OLS -------------------------------------------------------#
def pca(X,reqExp):
    X_ori = X
    pd.set_option('use_inf_as_na', True)
    X = np.log(((X /X.shift(1)).dropna(how='all')).astype(float))  # compute log returns
    X = np.array(X)
    # X = X[:math.floor(len(X) * w)] # use half of the data for PCA analysis (w = 0.5)
    cov_mat = np.cov(X.T) # Compute covariance matrix
    eigval, eigvec = np.linalg.eig(cov_mat) # Compute eigenvalues and eigenvectors
    eigval = np.real(eigval)
    eigvec = np.real(eigvec)
    n = np.where((np.cumsum(eigval)/sum(eigval))>=reqExp,0,1).sum()+1 # Get min number of eigenvectors to cover exp power
    print("Minimum number of eigenvectors to cover the required explanatory power: ",n)
    pc = eigvec.T[:n] # principal components
    # sigma = X.std() # compute volatility (standard deviation)
    # pc_adj = pc / np.array(sigma) # Adjust principal components by volatility
    # components = pd.DataFrame(np.dot(pc_adj, X_ori.T), columns=X_ori.index)
    components = pd.DataFrame(np.dot(pc, X_ori.T), columns=X_ori.index)

    return n, components

def standardise_return(stock,pc):
    stock = np.array(stock)
    pc = np.array(pc)
    stock_std = []
    pc_std = []
    for i in range(stock.shape[1]):
        stock_std.append((stock[:, i] - stock[:, i].mean()) / (stock[:, i].std()))  # standardize returns
    for i in range(pc.shape[0]):
        pc_std.append((pc[i] - pc[i].mean()) / (pc[i].std()))  # standardize returns

    return stock_std, pc_std

def fit_ols(stock,pc):
    X = stock
    stock,pc = standardise_return(stock,pc)
    # stock, pc = standardise_return(stock, pc)
    beta=[]
    residuals = pd.DataFrame()
    pc = pd.DataFrame(pc)
    pc = sm.add_constant(pc.T)
    for i in range(len(stock)):
        linreg = LinearRegression()
        model = linreg.fit(pc, stock[i])
        beta.append(model.coef_)
        residual = stock[i] - model.predict(pc)
        residuals[X.columns[i]] = residual

    cum_resid = residuals.cumsum()

    return beta, cum_resid
#------------------------------------------------- Strategy ----------------------------------------------------#
# 1. OU stochastic process: fit the residual of each stock
#     a. Multi-linear regression on the stock returns and the principal components to obtain betas and residuals.
#     b. Get parameters of OU-estimation on each stock, discrete OU process is equivalent to ARMA(1,0) model
# 2. Portfolio selection:
#     a. Get the stocks that have the fastest mean-conversion rate, and should be traded in the next level.
#     b. Get positions of each stocks in the whole trading periods.
#         - Compute trading signal of each stock
#         - Allocate funds on each stock that has a trading signal, whether long or short
def param_OUprocess(residual):
    SST = np.var(residual,axis=0)
    R_sq, k = [], []

    # Check stationarity before fitting ARMA(1,0) model
    # plot_acf(np.array(residual[0]))

    for i in range(residual.shape[1]):
        try:
            model = ARIMA(np.array(residual.iloc[:,i]).astype(float), order=(1,0,0))
            result = model.fit()
            SSE = result.resid.var()
            R_sq.append(1 - SSE / SST[i])
            phi = result.params[1]
            k.append(-np.log(phi))
        except:
            R_square.append(0.1)
            k.append(0.1)
            continue

    return R_sq,k

def portfolio_selection(data, pc, window=60, window_interval=5, n_sim=10):
    R_sq = []
    k = pd.DataFrame()
    n = data.shape[1]
    for i in range(n_sim):
        data_adj = data.iloc[n-window-1:n,:]
        num_pc, pc = pca(data_adj, reqExp=0.7)
        beta, residual = fit_ols(data_adj,pc)

        # get parameter k and R square
        R_sq_i, k_i = param_OUprocess(residual)
        R_sq.append(R_sq_i)
        k['k_%d'%i] = k_i
        n -= window_interval

    portfolio_index = k.mean(axis=1).sort_values(ascending=False)[:50].index # select top 50 stocks
    portfolio_index = data.columns[portfolio_index] # store ticker name

    return portfolio_index

def get_position(portfolio_index, stock_trade, pc, window=60, I=1):
    # return positions of each day with long/short signal
    # I denotes leverage, which should be set to minimize volatility of returns

    stock_data = stock_trade.loc[:, stock_trade.columns.isin(list(portfolio_index))]
    weighs,cum_residual_all = fit_ols(stock_data, pc)

    position = pd.DataFrame()
    sign_before = pd.DataFrame({"sign": [0] * len(stock_data.columns)}, index=stock_data.columns)

    for i in range(len(stock_data) - window):

        cum_residuals = cum_residual_all[:window+i]
        sign = trading_signal(cum_residuals, sign_before.sign, i) # use past T data to do OU fitting，and use T-1 data to generate trading signal
        n = len(sign)

        if i == 0:
            sign_before = sign
            sign_before['q'] = [0]*n
        if ((sign.sign >= 0).all() or (sign.sign <= 0).all()) and not (sign.sign == 0).all():
            pass
        elif (sign.sign == 0).all():
            sign_before.q = 0
        else:
            not_found = sign_before[~sign_before.index.isin(sign.index)]
            unchanged = np.dot(not_found.sign, not_found.q)
            I_adj = I - not_found.q.sum()

            q = optimize_allocation(sign, loadings, n, unchanged, I_adj) # compute optimization of portfolio
            sign['q'] = q
            sign_before = pd.concat([sign, not_found]) # if sign not changed, remain holding
        position[stock_data.index[i + window]] = sign_before.q * sign_before.sign
        print(i)

    position = position.T.sort_index()

    return position
#------------------------------------------- Backtesting --------------------------------------------------#
def trading_signal(cum_residuals, last_sign, i):
    # return signal，R_square，and signals
    # +1 : long
    # 0 : no change
    # -1 : short

    open_line = 1.75
    close_line = 0.5
    R_sq,k= param_OUprocess(cum_residuals)

    resid_T = cum_residuals[-1:]  # get cumulative residuals on the last day
    m = cum_residuals.mean()
    sigma = cum_residuals.std()

    adj_sigma = sigma / np.sqrt(k)
    signal = (resid_T - m) / adj_sigma
    signal = signal.T
    signal.columns = ['s']

    sign = pd.Series([0] * len(signal), name='sign', index=signal.index)

    if i != 0:
        sign.update(last_sign)

    sign = pd.concat([sign, ou_result['R_square'], signal], axis=1)

    open_short = sign['s'] > open_line
    open_long = sign['s'] < -open_line
    close_long = sign['s'] > -close_line
    close_short = sign['s'] < close_line
    current_long = last_sign > 0
    current_short = last_sign < 0

    if i != 0:
        # 1st trading day
        sign['sign'][current_long & close_long] = 0
        sign['sign'][current_short & close_short] = 0

    current_zero = sign['sign'] == 0
    sign['sign'][current_zero & open_long] = 1
    sign['sign'][current_zero & open_short] = -1

    # if R_square<0.75, clear position
    sign['sign'][sign['R_square'] < 0.75] = 0

    return sign


def optimize_allocation(sign, loadings, n, unchanged, I_adj = 1):
    # initialize bounds of positions q for n trading stocks,q>0
    bnds = [(0, None)] * n
    # initial values of each position is 0
    x0 = [0] * n

    def target_func(q, l, sign):
        target = l.apply(lambda x: abs(np.dot(x, sign * q))).sum()
        return target

    # arbitrage constraints
    def constraint_1(q, sign):
        return sum(sign * q) + unchanged

    # leverage constraints
    def constraint_2(q):
        return sum(q) - I_adj

    def constraint_3(q, sign):
        return abs((sign == 0)*q).sum()

    cons = ({'type': 'eq', 'fun': constraint_1, 'args': (sign.sign,)},
            {'type': 'eq', 'fun': constraint_2},
            {'type': 'eq', 'fun': constraint_3, 'args': (sign.sign,)})

    res = minimize(target_func, x0, method='SLSQP', bounds=bnds, constraints=cons, args=(loadings, sign.sign))

    return res.x

def earnings(stock_trade, position, epsilon=5e-4, window=60, r=0.04/252):
    # r denotes daily risk free rate
    # epsilon denotes trading fees

    stock_data = stock_trade[window:]
    stock_data = stock_data.loc[:, stock_data.columns.isin(position.columns)]

    ret = stock_data.fillna(method='ffill').pct_change().iloc[1:, :].fillna(0)

    borrow_rate = 0.0836 / 360
    cum_pnl = [1]
    wealth = 1 # initial wealth

    # returns from investment
    invest_return = wealth*pd.DataFrame(position.values[:-1] * ret.values, index=ret.index, columns=ret.columns).sum(axis=1)
    # initial investment cost
    invest_cost = r * position.sum(axis=1).values[:-1]
    # transaction cost
    transaction_cost = abs(position.diff()).sum(axis=1).values[1:] * epsilon
    # borrowing cost
    borrow_cost = abs(position[position < 0]).sum(axis=1).values[:-1] * borrow_rate
    total = invest_return - invest_cost - transaction_cost - borrow_cost

    for i in range(len(position) - 1):
        wealth += wealth * total[i]
        cum_pnl.append(wealth)

    cum_pnl = pd.Series(cum_pnl, index=position.index)

    ret_total = cum_pnl.pct_change().dropna()
    sharp_total = ((ret_total.mean())-r) / ret_total.std() * np.sqrt(252)

    return cum_pnl, sharp_total, total

#----------------------------------------------- Main -----------------------------------------------------#
def trade(trading_start):
    df_all, df_Est, df_Trade, df_vol = get_data(trading_start, window=60) # last price and volume of all tickers
    n, pc = pca(df_Est,reqExp=0.7)  # PCA analysis on the returns, default explanatory power is 0.7
    portfolio_index = portfolio_selection(df_Est, pc, window=60, window_interval=5, n_sim=10)
    bet = get_position(portfolio_index,df_Trade,pc,window,I=1)
    cum_pnl, sharp_total, total = earnings(df_Trade, bet)
    plt.plot(cum_pnl)

    return sharp_total

if __name__ == '__main__':
    time_list = ['2015-01-04', '2016-01-04', '2017-01-04', '2018-01-04', '2019-01-04',
                 '2019-01-04', '2020-01-04','2021-01-04']
    sharp_all = []
    for i in time_list:
        sharp = trade(i)
        sharp_all.append(sharp)
    print(sharp_all)

