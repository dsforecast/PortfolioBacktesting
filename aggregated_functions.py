# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:31:14 2021

@author: Frey
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import truncnorm
from scipy.special import betainc


def mdd(data):
    ''' mdd: Returns maximum drawdown for return series
    '''
    try:
        # i = (np.maximum.accumulate(data)-data).values.argmax()
        # j = (data[:i]).values.argmax()
        # mdd0 = data.iloc[j]-data.iloc[i]
        mdd0 = -np.min((1+data).cumprod(axis=0)/(1+data).cumprod(axis=0).cummax(axis=0)-1,axis=1)
    except (RuntimeError, ValueError, TypeError, NameError, IndexError):
        mdd0 = 0
    return mdd0
def annual_measures(returns):
    ''' annual_measures: Returns annual performance measures for return series
    '''
    length_year = 252
    returns = returns.dropna()
    all_years = [x for x in range(returns.index[0].year, returns.index[-1].year+1, 1)]
    annuals = np.zeros((len(all_years), 6))
    for i in enumerate(all_years):
        annuals[i[0], 0] = ((1+returns[str(i[1])]).cumprod().tail(1)-1)*100
        annuals[i[0], 1] = returns[str(i[1])].mean()*length_year*100
        annuals[i[0], 2] = (returns[str(i[1])].std()*length_year**.5)*100
        annuals[i[0], 3] = mdd(returns[str(i[1])])*100
        annuals[i[0], 4] = annuals[i[0], 1]/annuals[i[0], 2]
        annuals[i[0], 5] = annuals[i[0], 1]/annuals[i[0], 3]
    annuals = pd.DataFrame(annuals)
    annuals.columns = ['TotalReturn', 'MeanReturn', 'Volatility', 'MDD', 'Sharpe', 'Calmar']
    annuals.index = all_years
    return annuals

def total_measures(returns, length_year, gamma):
    '''
    total_measures: Calculate total performance measures for returns
    '''
    totals = pd.DataFrame(columns = returns.columns)
    totals.loc['TotalReturn',:] = (((1+returns).cumprod(axis=0).tail(1)-1)*100).values
    totals.loc['MeanReturn',:] = (returns.mean(axis=0)*length_year*100).values
    totals.loc['Volatility',:] = ((returns.std(axis=0)*length_year**.5)*100).values
    totals.loc['MDD',:] = mdd(returns.T) * 100
    totals.loc['Sharpe',:] = [(totals.loc['MeanReturn',i] / totals.loc['Volatility',i]) if totals.loc['Volatility',i] != 0 else 0 for i in totals.columns]
    totals.loc['Calmar',:] = [(totals.loc['MeanReturn',i] / totals.loc['MDD',i]) if totals.loc['MDD',i] != 0 else 0 for i in totals.columns]
    totals.loc['Skewness',:] = ((returns.skew(axis=0)*length_year**.5)*100).values
    totals.loc['Kurtosis',:] = ((returns.kurt(axis=0)*length_year**.5)*100).values
    totals.loc['Return Loss'] = totals.loc['Volatility',:].mul(totals.loc['Sharpe','1/N'],axis=0) - totals.loc['MeanReturn',:]
    totals.loc['Certainty Equivalent'] = totals.loc['MeanReturn',:] - gamma/2 * totals.loc['Volatility',:]
    for i in enumerate(totals.index):
        totals.loc[i[1],:] = round((totals.loc[i[1],:].astype(float)), 2)
    return totals

def capture_ratio(data):
    '''
    capture_ratio: calculate return ratio upside to downside betwenn bm and strat
    '''
    data.columns = ['benchmark', 'strat']
    def calculate_ratio(returns):
        '''
        calculate_ratio: calculate return ratio betwenn bm and strat
        '''
        multiplier = abs(returns.loc[:,'benchmark'])/abs(returns.loc[:,'benchmark']).sum()
        ratio = (returns.loc[:,'strat']/returns.loc[:,'benchmark'])
        ratio = (ratio*multiplier).sum()
        return ratio
    ratio_upside = calculate_ratio(data.loc[data.loc[:,'benchmark']>0])
    ratio_downside = calculate_ratio(data.loc[data.loc[:,'benchmark']<0])
    return float(ratio_upside / ratio_downside)

def turnover_measures(allocation, dates):
    ''' turnover_measures: Returns turnover measures for allocation
    '''
    # annual turnover
    allocation = allocation.dropna()
    weight_change = abs(allocation-allocation.shift()).sum(axis=1)
    weight_change[0] = 1
    tmp_to = []
    for i in enumerate(dates):
        tmp_to.append(weight_change[dates[i[0]]]*100)
    annual_turnover = pd.DataFrame(tmp_to)
    annual_turnover['dates'] = dates
    annual_turnover['year'] = pd.to_datetime(annual_turnover['dates'], format = '%Y-%m-%d').dt.strftime('%Y')
    annual_turnover = annual_turnover.groupby(['year']).sum()
    # turnover
    turnover_new = pd.DataFrame(index=allocation.index, columns=allocation.columns).fillna(0)
    turnover_new[allocation!=0]=1
    diff_turnover_new=turnover_new.diff()
    diff_turnover_new.iloc[0,:] = turnover_new.iloc[0,:]
    turnover_news = (diff_turnover_new==1).astype(int).sum(axis=1).replace(to_replace=0, method='ffill')
    turnover_olds = (diff_turnover_new==-1).astype(int).sum(axis=1).replace(to_replace=0, method='ffill')
    turnover_prob = (turnover_news + turnover_olds)/turnover_new.astype(int).sum(axis=1)*100
    turnover_prob = turnover_prob.where(~turnover_prob.duplicated(),0)[:-1]
    turnover_news = turnover_news[1:]
    turnover_olds = turnover_olds[1:]
    turnover = allocation - allocation.shift(1)
    turnover.dropna(how='all', inplace=True)
    turnover.fillna(value=0, inplace=True)
    turnover_value = turnover.abs().sum(axis=1)
    turnover_count = turnover.copy()
    turnover_count[turnover_count != 0] = 1
    turnover_count = turnover_count.sum(axis=1)
    turnover_count_in = turnover.copy()
    turnover_count_in[turnover_count_in > 0] = 1
    turnover_count_in = round(turnover_count_in.sum(axis=1))
    turnover_count_out = turnover.copy()
    turnover_count_out[turnover_count_out < 0] = 1
    turnover_count_out = round(turnover_count_out.sum(axis=1))
    allocation.dropna(how='all', inplace=True)
    turnover_constituents = allocation.div(allocation).fillna(0).sum(axis=1)#pd.Series(np.count_nonzero(allocation, axis=1))
    turnover_constituents = turnover_constituents[turnover_value.index] #turnover_constituents[1:]
    turnover_constituents.index = turnover_value.index

    turnover = pd.concat([turnover_value, turnover_count, turnover_count_in,\
                          turnover_count_out, turnover_constituents, turnover_news, turnover_olds, turnover_prob], axis=1)
    turnover.columns = ['turnover_value', 'turnover_count', 'turnover_count_in',\
                        'turnover_count_out', 'turnover_constituents', 'turnover_news', 'turnover_olds', 'turnover_prob']
    return turnover, annual_turnover

def hierarchical_ridge(y,X,B):
    '''
    Parameters
    ----------
    y : array, float
        (T x 1) column vector of left-handside centered variable.
    X : array, float
        (T x N) matrix of right-handside centered variables.
    B : int
        Scalar, number of Gibbs sampler iterations.
    Returns
    -------
    beta = (B x N) posterior draws of regression coefficients
    sigma = (B x 1) posterior draws of error variance
    invtau2 = (B x N) posterior draws of 1/t_j^2
    lambda = (B x 1) posterior draws of lambda
    Reference: Bauwens, L. and D. Korobilis (2013): Bayesian methods, in Handbook of Research
               Methods And Applications In Empirical Macroeconomics, ed. by N. Hashimzade and M. A.
               Thornton, Edward Elger Publishing, Handbooks of Research Methods and Applications
               Series, chap. 16, 363{380.
    '''
    # Dimensions
    T, N = X.shape

    # MCMC Set-Up
    _burnin = math.ceil(0.3*B)
    _ndraw = B + _burnin
    beta = np.zeros([B,N])
    sigma = np.zeros([B,1])
    tau = np.zeros([B,N])

    # OLS quantities
    _betaOLS = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    _SSE = (y-X @ _betaOLS).T @ (y-X @ _betaOLS)
    _sigmaOLS = _SSE/(T-N*(N<T))

    # Parameter values for first Gibbs sammple step
    betaD = _betaOLS
    tauD = np.zeros([1, N])
    sigmaD = 1*(T<N) + _sigmaOLS*(T>N)

    # Further hyperparameters
    b0 = np.zeros([N, 1])
    v0 = 0.0001
    v1 = T/2 + v0/2
    s0 = 0.1 # IG_2(a,b) = IG(a/2,b/2)
    q1 = 0.0001
    q2 = 0.0001

    # MCMC
    for i in range(_ndraw):

        # (1) Sample tau | beta
        for j in range(N):
            tauD[0, j] = 1/np.random.gamma(q1+1, q2+np.square(betaD[j]))

        # (2) Sample sigma | beta
        # Note that no term with M0 enters because the prior variance of beta is independent of 1/sigma^2.
        s1 = (y-X @betaD).T @ (y-X @betaD)/2 + s0
        sigmaD =1/(np.random.gamma(v1, 1/s1))

        # (3) Sample beta | tau, sigma
        M0 = np.diag(tauD[0])
        M0b0 = np.linalg.pinv(M0) @ (b0)
        M1 = np.linalg.pinv(M0) + (X.T @ X)/sigmaD
        invM1 = np.linalg.pinv(M1)
        b1 = invM1 @ (M0b0 + (X.T @ y)/sigmaD)
        try:
            betaD = np.array(np.random.multivariate_normal(b1.squeeze(), invM1))
            kickout = False
        except:
            betaD = np.array(np.random.multivariate_normal(b1.squeeze(), np.eye(N)))
            kickout = True
        betaD = np.expand_dims(betaD, axis=1)

        # Save everything
        if (i>_burnin) and (kickout==False):
            beta[i-_burnin,:] = betaD.squeeze()
            sigma[i-_burnin, 0] = sigmaD.squeeze()
            tau[i-_burnin,:] = tauD.squeeze()

    return beta, sigma, tau

def bayesian_lasso(y,X,B):
    '''
    Parameters
    ----------
    y : array, float
        (T x 1) column vector of left-handside centered variable.
    X : array, float
        (T x N) matrix of right-handside centered variables.
    B : int
        Scalar, number of Gibbs sampler iterations.
    Returns
    -------
    beta = (B x N) posterior draws of regression coefficients
    sigma = (B x 1) posterior draws of error variance
    invtau2 = (B x N) posterior draws of 1/t_j^2
    lambda = (B x 1) posterior draws of lambda
    Reference: Park, T. and Casella, G. (2008): "The Bayesian Lasso,"
               Journal of American Statistical Association, 103(482), 681--686.
    '''
    # Dimensions
    T, N = X.shape

    # MCMC Set-Up
    _burnin = math.ceil(0.3 * B)
    _ndraw = B + _burnin
    beta = np.zeros([B,N])
    sigma = np.zeros([B,1])
    invtau2 = np.zeros([B,N])
    lambda_out = np.zeros([B,1])

    # OLS quantities
    _betaOLS = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    _SSE = (y-X @ _betaOLS).T @ (y-X @ _betaOLS)
    _sigmaOLS = _SSE/(T-N*(N<T))

    # Parameter values for first Gibbs sammple step
    sigmaD = 1*(T<N) + _sigmaOLS*(T>N)
    invtau2D = (1 / np.square(_betaOLS).T) * (T>N)+ 1 * (T<=N)
    lambdaD = 0.1

    # Further hyperparameters
    r = 0.001
    delta = 0.001

    # MCMC
    for i in range(_ndraw):
        # (1) Sample beta | tau, sigma
        invD = np.diag(invtau2D[0])
        try:
            invA = np.linalg.pinv(X.T @ X + invD)
            betaD = np.array(np.random.multivariate_normal(invA @ X.T @ y.T[0],sigmaD * invA))
            kickout = False
        except:
            betaD = np.random.multivariate_normal(invA @ X.T @ y.T[0],np.eye(N))
            kickout = True
        betaD = np.expand_dims(betaD, axis=1)

        # (2) Sample sigma | beta, tau
        s = (y-X @betaD).T @ (y-X @betaD)/2 + betaD.T @ invD @ betaD/2
        sigmaD = 1/(np.random.gamma((T+N-1)/2, 1/s))

        # (3) Sample invtau2 | sigma, beta
        for j in range(N):
            mu = np.sqrt(lambdaD*sigmaD /np.square(betaD[j]))
            invtau2D[0, j] = np.random.wald(mu, lambdaD)

        # (4) Sample lambda form Gamma | tau
        s2 = (1/invtau2D).sum()/2 + delta
        lambdaD = np.random.gamma(N+r, 1/s2)

        # Save everything
        if (i>_burnin) and (kickout==False):
            beta[i-_burnin,:] = betaD.squeeze()
            sigma[i-_burnin, 0] = sigmaD.squeeze()
            invtau2[i-_burnin,:] = invtau2D.squeeze()
            lambda_out[i-_burnin, 0] = lambdaD

    return beta, sigma, invtau2, lambda_out

def bayesian_elastic_net(y,X,B):
    '''
    Parameters
    ----------
    y : array, float
        (T x 1) column vector of left-handside centered variable.
    X : array, float
        (T x N) matrix of right-handside centered variables.
    B : int
        Scalar, number of Gibbs sampler iterations.
    Returns
    -------
    beta = (B x N) posterior draws of regression coefficients
    sigma = (B x 1) posterior draws of error variance
    invtau2 = (B x N) posterior draws of 1/t_j^2
    lambda1 = (B x 1) posterior draws of lambda
    lambda2 = (B x 1) posterior draws of lambda2
    Reference: Kyung, M., Gill, J., Ghosh, M. and Casella, G. (2010):
               "Penalized Regression, Standard Errors, and Bayesian Lassos,"
               Bayesian Analysis, 5(2), 369--412.
    '''
    # Dimensions
    T, N = X.shape

    # MCMC Set-Up
    _burnin = math.ceil(0.3 * B)
    _ndraw = B + _burnin
    beta = np.zeros([B,N])
    sigma = np.zeros([B,1])
    invtau2 = np.zeros([B,N])
    lambda1_out = np.zeros([B,1])
    lambda2_out = np.zeros([B,1])

    # OLS quantities
    _betaOLS = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    _SSE = (y-X @ _betaOLS).T @ (y-X @ _betaOLS)
    _sigmaOLS = _SSE/(T-N*(N<T))

    # Parameter values for first Gibbs sammple step
    sigmaD = 0.01*(T<N) + _sigmaOLS*(T>N)
    invtau2D = (1 / np.square(_betaOLS).T) * (T>N)+ 0.01 * (T<=N)
    lambdaD1 = 1
    lambdaD2 = 1

    # Further hyperparameters
    r1 = 0.001
    r2 = 0.001
    delta1 = 0.001
    delta2 = 10.001

    # MCMC
    for i in range(_ndraw):
        # (1) Sample beta | tau, sigma
        invD = np.diag(invtau2D[0]+lambdaD2)
        try:
            invA = np.linalg.pinv(X.T @ X + invD)
            betaD = np.array(np.random.multivariate_normal(invA @ X.T @ y.T[0],sigmaD * invA))
            kickout = False
        except:
            betaD = np.random.multivariate_normal(invA @ X.T @ y.T[0],np.eye(N))
            kickout = True
        betaD = np.expand_dims(betaD, axis=1)

        # (2) Sample sigma | beta, tau
        s = (y-X @betaD).T @ (y-X @betaD)/2 + betaD.T @ invD @ betaD/2
        sigmaD = 1/(np.random.gamma((T+N-1)/2, 1/s))

        # (3) Sample invtau2 | sigma, beta
        for j in range(N):
            mu = np.sqrt(lambdaD1*sigmaD /np.square(betaD[j]))
            invtau2D[0, j] = np.random.wald(mu, lambdaD1)

        # (4) Sample lambda1, lambda2 from Gamma | tau
        s12 = (1/invtau2D).sum()/2 + delta1
        lambdaD1 = np.random.gamma(N+r1, 1/s12)
        s22 = (1/invtau2D).sum()/2 + delta1
        s22 = np.square(betaD).sum()/(2*sigmaD) + delta2
        lambdaD2 = np.random.gamma(N+r2, 1/s22)[0][0]

        # Save everything
        if (i>_burnin) and (kickout==False):
            beta[i-_burnin,:] = betaD.squeeze()
            sigma[i-_burnin, 0] = sigmaD.squeeze()
            invtau2[i-_burnin,:] = invtau2D.squeeze()
            lambda1_out[i-_burnin, 0] = lambdaD1
            lambda2_out[i-_burnin, 0] = lambdaD2

    return beta, sigma, invtau2, lambda1_out, lambda2_out

def truncted_normal(y,X,B, lower_bound, upper_bound):
    '''
    Parameters
    ----------
    y : array, float
        (T x 1) column vector of left-handside variable
    X : array, float
        (T x K) matrix of right-handside variables
    B : scalar, int
        number of Gibbs sampler iterations
    lower_bound: scalar, float
        lower truncation bound
    upper_bound: scalar, float
        upper truncation bound
    Returns
    -------
    beta = (B x N) posterior draws of regression coefficients
    sigma = (B x 1) posterior draws of error variance
    Reference: [1] Rodriguez-Yam, G., Davis, R.A. and and Scharf L.L. (2014): "Efficient
                   Gibbs Sampling of Truncated Multivariate Normal with Application
                   to Constrained Linear Regression, Wokring Paper
               [2] Geweke, J. (1996): "Bayesian Inference for Linear Models Subject to
                   Linear Inequality Constraints," in Modeling and Prediction: Honouring
                   Seymour Geisser, eds. W. O. Johnson, J. C. Lee, and A. Zellner,
                   New York, Springer, pp. 248-263.
    '''
    # Dimensions
    T, N = X.shape

    # MCMC Set-Up
    _burnin = math.ceil(0.3 * B)
    _ndraw = B + _burnin
    beta = np.zeros([B,N])
    sigma = np.zeros([B,1])
    index_vec = np.array([*range(N)])
    a = lower_bound * np.ones([N,1])
    b = upper_bound * np.ones([N,1])
    H = np.eye(N)

    # OLS quantities
    _betaOLS = np.linalg.pinv(X.T @ (X)) @ (X.T @ y)
    _SSE = (y-X @ _betaOLS).T @ (y-X @ _betaOLS)
    _sigmaOLS = _SSE/(T-N*(N<T))

    # Parameter values for first Gibbs sammple step
    sigmaD = 0.01*(T<N) + _sigmaOLS*(T>N)
    betaD = _betaOLS
    gammaD = _betaOLS

    # MCMC
    for i in range(_ndraw):

        # (1) Draw gamma=H*beta
        omega = np.linalg.pinv(X.T @ X) * sigmaD + np.eye(N)
        gammabar = np.linalg.pinv(omega.T) @ (X.T @ y/sigmaD)
        for j in range(N):
            whole_vec = omega[j,:].T * (gammaD-gammabar)
            points = np.where(index_vec!=j)[0][0]
            vec_keep = whole_vec[points]
            mean_part = gammabar[j] - vec_keep.sum()/omega[j,j]
            mean_part = max(a[j],min(b[j],mean_part))
            var_part = 1/omega[j,j]
            gamma_draw = truncnorm.rvs(a[j], b[j]) * np.sqrt(var_part) + mean_part
            gammaD[j] = gamma_draw

        betaD = np.linalg.pinv(H) @ gammaD
        # (2) Draw sigma^2
        s = (y-X @ gammaD).T @ (y-X @ gammaD)/2
        sigmaD = 1/np.random.gamma(T/2, 1/s);

        # Save everything
        if (i>_burnin):
            beta[i-_burnin,:] = betaD.squeeze()
            sigma[i-_burnin, 0] = sigmaD.squeeze()

    return beta, sigma

def ledoit_wolf(data, method):
    '''
    Parameters
    ----------
    data : array, float
        (T x N) matrix of asset returns
    method : scalar, int
             method = 1, shrinkage towards single-index covariance matrix
                    = 2, shrinkage towards sigma2 * I_(N x N)
                    = 3, shrinkage towards constant correlation matrix
    Returns
    -------
    Sigma : array, float
        (N x N) matrix, invertible covariance matrix estimator
    weights : array, float
        (1 x N) vector of portfolio weights
    Reference
    ---------
    [1] Ledoit, O. and Wolf, M. (2003): "Improved estimation of the
        covariance matrix of stock returns with an application to
        portfolio selection," Journal of Empirical Finance, 10(5), 603-–621
    [2] Ledoit, O. and Wolf, M. (2004): "A well-conditioned estimator for
        large-dimensional covariance matrices," Journal of Multivariate
        Analysis, 88(2), 365--411
    [3] Ledoit, O. and Wolf, M. (2004): "Honey, I Shrunk the Sample
        Covariance Matrix," The Journal of Portfolio Management, 30(4), 110--119
    Disclaimer
    ----------
    Code is taken and adapted from https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html
    '''
    # Array structure
    returns = np.array(data)#

    # Dimensions
    T, N = returns.shape

    # Sample variance-covariance matrix
    S = np.cov(returns.T)

    if method == 1:
        market = returns.mean(axis=1)[:, np.newaxis]
        sample = np.cov(np.concatenate((returns, market), axis=1).T)
        covmkt = sample[:N,N, np.newaxis]
        varmkt = sample[N, N]
        sample = sample[1:N+1, 1:N+1]
        F = covmkt @ covmkt.T/varmkt
        np.fill_diagonal(F,np.diag(sample))
    	# compute shrinkage parameters
        c = np.square(np.linalg.norm(sample-F, ord='fro'))
        y = np.square(returns)
        p = 1/T * np.sum(y.T @ y) - np.sum(np.square(sample))
        # r is divided into diagonal and off-diagonal terms,
        # and the off-diagonal term is itself divided into smaller terms
        rdiag = 1/T * np.sum(np.square(y)) - np.sum(np.square(np.diag(sample)))
        z = returns * market
        v1 = 1/T * y.T @ z - covmkt * sample
        roff1 = np.sum(v1 * covmkt.T)/varmkt - np.sum(np.diag(v1) * np.square(covmkt))/np.square(varmkt)
        v3 = 1/T * z.T @ z - varmkt * sample
        roff3 = np.sum(v3.dot(covmkt @ (covmkt.T)))/np.square(varmkt) - np.sum(np.diag(v3) * np.square(covmkt))/np.square(varmkt)
        roff = 2 * roff1 - roff3
        r = rdiag + roff

    if method == 2:
        sample = S
        meanvar = np.mean(np.diag(sample))
        F = meanvar * np.eye(N)
    	# compute shrinkage parameters
        y = np.square(returns)
        phiMat = y.T @ y/T -np.square(sample)
        p = np.sum(phiMat)
        r = 0
        c = np.square(np.linalg.norm(sample-F, ord='fro'))

    if method == 3:
        sample = S
        # compute F
        var = np.diag(sample)
        sqrtvar = np.sqrt(var)[:, np.newaxis]
        rBar = np.sum(sample.dot(1/(sqrtvar.dot(sqrtvar.T)))-N)/(N*(N-1))
        F = rBar * sqrtvar.dot(sqrtvar.T)
        np.fill_diagonal(F,var)

        # compute shrinkage parameters
        y = np.square(returns)
        phiMat = y.T @ y/T -np.square(sample)
        p = np.sum(phiMat)

        # what we call rho-hat
        term1 = np.power(returns, 3).T @ returns/T
        helpf = returns.T @ (returns)/T
        helpDiag = np.diag(helpf)
        term2 = helpDiag.dot(sample)
        term3 = helpf.dot(var)
        term4=var.dot(sample)
        thetaMat = term1 - term2 - term3 + term4
        np.fill_diagonal(thetaMat,np.zeros([N, 1]))
        r = np.sum(np.diag(phiMat)) + rBar * np.sum(((1./sqrtvar) @ sqrtvar.T).dot(thetaMat))

    	# what we call gamma-hat
        c = np.square(np.linalg.norm(sample-F, ord='fro'))

    # Compute shrinkage constant
    kappa = (p-r)/c
    shrinkage = max(0,min(1,kappa/T))

    # Calculate invertible covariance matrix estimator
    Sigma = shrinkage * F + (1-shrinkage) * S

    # Calculate Sigma^-1
    invSigma = np.linalg.pinv(Sigma)
    # invSigma = Sigma ** -1

    # Portfolio Weights
    weights = invSigma @ np.ones([N,1]) / (np.ones([N,1]).T @ invSigma @ np.ones([N,1]))

    # Robustness:
    if abs(weights).sum() > N:
        invSigma = Sigma ** -1
        weights = invSigma @ np.ones([N,1]) / (np.ones([N,1]).T @ invSigma @ np.ones([N,1]))

    return weights, Sigma

def frahm_memmel(returns, kappa_method):
    '''
    Parameters
    ----------
    returns : array, float
        (T x N) matrix of asset returns
    kappa_method : scarlar, int
            = 1, if kappa as in equation (8)
            = 2, if kappa as in equation (10)
    Returns
    -------
    weights : array, float
        (1 x N) vector of GMV portfolio weights
    Reference
    ---------
    Frahm, G. and C. Memmel (2010): Dominating estimators for minimum-variance
    portfolios, Journal of Econometrics, 159, 289-302.
    '''
    # Array structure
    returns = np.array(returns)

    # Dimensions
    T, N = returns.shape

    # Parameter Calculations
    Sigma = np.cov(returns.T)
    invSigma = np.linalg.pinv(Sigma)
    omegaR = np.ones([N,1])/N
    omegaT = invSigma @ np.ones([N,1]) / (np.ones([N,1]).T @ invSigma @ np.ones([N,1]))
    sigmaR = omegaR.T @ Sigma @ omegaR
    sigmaT = omegaT.T @ Sigma @ omegaT
    tauR = (sigmaR-sigmaT)/sigmaR

    # Define kappa
    kappa = (N-3)/((T-N+2)*tauR)
    if kappa_method == 2:
        kappa = min(kappa, 1)

    # Calculate weights
    weights = kappa * omegaR + (1-kappa) * omegaT

    return weights

def tou_zhou(returns, gamma):
    '''
    Parameters
    ----------
    returns : array, float
        (T x N) matrix of asset returns
    gamma : scarlar, int
        risk aversion parameter
    Returns
    -------
    weights : array, float
        (1 x N) vector of GMV portfolio weights
    Reference
    ---------
    Tu, J. and G. Zhou (2011): Markowitz meets Talmud: A combination of sophisticated and naive
    diversification strategies, Journal of Financial Economics, 99, 204-215.
    '''
    # Array structure
    returns = np.array(returns)

    # Dimensions
    T, N = returns.shape

    # Parameter Calculations
    Sigma = np.cov(returns.T)
    invSigma = np.linalg.pinv(Sigma)
    w_gmvp = invSigma @ np.ones([N,1]) / (np.ones([N,1]).T @ invSigma @ np.ones([N,1]))
    mu_min = np.mean(returns, axis=0) @ w_gmvp
    hat_psi2 = (np.mean(returns, axis=0)-mu_min) @ invSigma @ (np.mean(returns, axis=0)-mu_min).T
    hat_psi2a = ((T-N-1)*hat_psi2-(N-1))/T+(2*np.power(hat_psi2,(N-1)/2) * np.power(1+hat_psi2,-(T-2)/2))/(T*betainc((N-1)/2,(T-N+1)/2,hat_psi2/(1+hat_psi2)))
    c1 = (T-2) * (T-N-2) / ((T-N-1) * (T-N-4))
    hat_pi1 = np.ones([1,N])/N  @ invSigma @ np.ones([N,1])/N - 2/gamma * np.ones([1,N])/N @ np.mean(returns, axis=0).T + hat_psi2a/np.square(gamma)
    hat_pi2 = hat_psi2a*(c1-1)/(gamma**2)+c1/(gamma**2)*N/T
    hat_delta = hat_pi1/(hat_pi1 + hat_pi2)

    # Portfolio weights
    weights0 = (1-hat_delta) * np.ones([1,N])/N + hat_delta * invSigma @ np.mean(returns, axis=0)/gamma
    weights = weights0.squeeze()[:, np.newaxis]

    return weights


def fama_french(returns, factors):
    '''
    Parameters
    ----------
    weights : TYPE
        DESCRIPTION.

    Returns
    -------
    weights : array, float
        (1 x N) vector of GMV portfolio weights
    Reference
    ---------
    Fama, E. F. and K. R. French (2015): A five-factor asset pricing model,
    Journal of Financial Economics, 116, 1-22.
    '''
    # Array structure
    returns = np.array(returns)
    factors = np.array(factors)[:,:5]

    # Dimensions
    T, N = returns.shape

    # Parameter Calculations
    y = np.mean(returns, axis=1)[:, np.newaxis]
    x = factors
    b = np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
    res = y - x @ b
    Sigma = b @ (np.cov(factors.T) @ b).T + np.cov(res.T) * np.eye(N)
    invSigma = np.linalg.pinv(Sigma)

    # Portfolio Weights
    weights = invSigma @ np.ones([N,1]) / (np.ones([N,1]).T @ invSigma @ np.ones([N,1]))

    return weights
