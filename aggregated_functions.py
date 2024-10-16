# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:31:14 2021

@author: Frey
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import truncnorm, norm, f
from scipy.special import betainc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def shift_intervall(data, lower_bound, upper_bound):
    '''
    Parameters
    ----------
    data : array, float
    lower_bound : float
    upper_bound : float
    Returns
    -------
    shifted_data : array, float
        data shifted to intervall [lower_bound, upper_bound]
    '''
    a, b = data.min(), data.max()
    c, d = lower_bound, upper_bound
    shifted_data = data.copy()
    if any(shifted_data<0):
        shifted_data = c + (d - c) / (b - a) * (shifted_data - a) #c + (d - c) / ((b - a) * (a < b) + (a >= b)) * (data - a)
    return shifted_data

def mdd(data):
    ''' mdd: Returns maximum drawdown for return series
    '''
    try:
        mdd0 = -np.min((1+data).cumprod(axis=0)/(1+data).cumprod(axis=0).cummax(axis=0)-1,axis=0)
    except (RuntimeError, ValueError, TypeError, NameError, IndexError):
        mdd0 = 0
    return mdd0
def annual_measures(returns):
    ''' annual_measures: Returns annual performance measures for return series
    '''
    returns = returns#.dropna()
    all_years = [x for x in range(returns.index[0].year, returns.index[-1].year+1, 1)]
    annuals = np.zeros((len(all_years), 6))
    for i in enumerate(all_years):
        annuals[i[0], 0] = ((1+returns[str(i[1])]).cumprod().tail(1)-1)*100
        annuals[i[0], 1] = returns[str(i[1])].mean()*100
        annuals[i[0], 2] = returns[str(i[1])].std()*100
        annuals[i[0], 3] = mdd(returns[str(i[1])].T)*100
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
    totals.loc['MDD',:] = mdd(returns) * 100
    totals.loc['Sharpe',:] = [(totals.loc['MeanReturn',i] / totals.loc['Volatility',i]) if totals.loc['Volatility',i] != 0 else 0 for i in totals.columns]
    totals.loc['Calmar',:] = [(totals.loc['MeanReturn',i] / totals.loc['MDD',i]) if totals.loc['MDD',i] != 0 else 0 for i in totals.columns]
    totals.loc['Skewness',:] = ((returns.skew(axis=0)*length_year**.5)*100).values
    totals.loc['Kurtosis',:] = ((returns.kurt(axis=0)*length_year**.5)*100).values
    totals.loc['Return Loss'] = totals.loc['Volatility',:].mul(totals.loc['Sharpe','1/N'],axis=0) - totals.loc['MeanReturn',:]
    totals.loc['Certainty Equivalent'] = totals.loc['MeanReturn',:] - gamma/2 * totals.loc['Volatility',:]
    return totals

def turnover_measures(allocation, dates):
    ''' turnover_measures: Returns turnover measures for allocation
    '''
    # annual 
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
    turnover = allocation - allocation.shift(1)
    turnover.dropna(how='all', inplace=True)
    turnover.fillna(value=0, inplace=True)
    turnover = turnover.abs().sum(axis=1)
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
    s0 = 1#1.1 # IG_2(a,b) = IG(a/2,b/2)
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

def bayesian_lasso(y,X,B, r=0.0001, delta=0.0001):
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
    # r = 0.001
    # delta = 0.001

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
    
    if T <= N:
        return np.full((N,), np.nan)

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
    
    if T <= N:
        return np.full((N,), np.nan)

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
    factors = np.array(factors.iloc[:,:-1])

    # Dimensions
    T, N = returns.shape

    # Parameter Calculations
    y = returns
    x = factors
    b = np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
    res = y - x @ b
    Sigma = b.T @ (np.cov(factors.T) @ b) + np.cov(res.T) * np.eye(N)
    invSigma = np.linalg.pinv(Sigma)

    # Portfolio Weights
    weights = invSigma @ np.ones([N,1]) / (np.ones([N,1]).T @ invSigma @ np.ones([N,1]))

    return weights


# In[Latex Output]

# Function to assign asterisks based on p-values
def add_asterisks(p):
    thresholds = [(0.01, '***'), (0.05, '**'), (0.1, '*')]
    for threshold, symbol in thresholds:
        if p <= threshold:
            return symbol
    return ''

# Function to get the value to highlight (max, min, or mid)
def get_highlight_value(s, highlight_type):
    if highlight_type == 'max':
        return s.max()
    elif highlight_type == 'min':
        return s.min()
    elif highlight_type == 'mid':
        return s.median()  # Mid value is considered the median
    else:
        raise ValueError("Invalid highlight_type. Use 'max', 'min', or 'mid'.")
        
# Function to apply \textbf{} to column headers only
def boldify_headers(df):
    df_bold = df.copy()
    
    # Apply bold formatting to columns (MultiIndex if applicable)
    if isinstance(df_bold.columns, pd.MultiIndex):
        # Rebuild the MultiIndex with bold formatting applied to all levels
        df_bold.columns = pd.MultiIndex.from_tuples(
            [tuple(f"\\textbf{{{level}}}" for level in col) for col in df_bold.columns]
        )
    else:
        df_bold.columns = [f"\\textbf{{{col}}}" for col in df_bold.columns]

    return df_bold

# Function to apply formatting, bold for max, and asterisks based on p-values
def highlight_max_with_pvals(df, pvals=None, highlight_type='max', decimal_format='.2f', bold=False):
    # If pvals is not provided, set it to a DataFrame of ones with the same shape as df
    if pvals is None:
        pvals = pd.DataFrame(1, columns=df.columns, index=df.index)
    
    # # Function to format each column
    # def format_column(s, pvals_col):
    #     highlight_val = get_highlight_value(s, highlight_type)  # Get value to highlight (max, min, or mid)

    #     formatted_values = []
        
    #     for v, p in zip(s, pvals_col):
    #         value_str = f"{v:{decimal_format}}{add_asterisks(p)}"  # Apply decimal format and asterisks
    #         if v == highlight_val:  # Bold the maximum value
    #             value_str = f"\\cellcolor{{gray!30}}\\textbf{{{value_str}}}"
    #         formatted_values.append(value_str)
        
    #     return formatted_values
    
    # Function to create shading based on rank
    def format_column(s, pvals_col):
        
        highlight_val = get_highlight_value(s, highlight_type)  # Get value to highlight (max, min, or mid)
        if highlight_type == 'max':
            ranks = s.rank(ascending=False, numeric_only=True)  # Get the rank of values in the column
        else:
            ranks = s.rank(ascending=True, numeric_only=True)
        max_rank = len(s)  # Maximum possible rank
        
        formatted_values = []
    
        for v, r, p in zip(s, ranks, pvals_col):
            # Calculate the intensity of the shading (higher rank = darker shade)
            if np.isnan(r):
                r = max_rank
            # if highlight_type == 'max':
            shade_intensity = int((1-r/max_rank) * 50)  # Scale rank to 0-90 for gray! shading
            # else:
            #     shade_intensity = int((r/max_rank) * 50)  # Scale rank to 0-90 for gray! shading
            value_str = f"{v:{decimal_format}}{add_asterisks(p)}"  # Apply decimal format and asterisks
            
            if v == highlight_val:  # Bold the maximum value
                value_str = f"\\textbf{{{value_str}}}"
            
            # Apply cell shading based on the rank
            value_str = f"\\cellcolor{{gray!{shade_intensity}}}{value_str}"
            formatted_values.append(value_str)
        
        return formatted_values
    
    return_df = pd.DataFrame({col: format_column(df[col], pvals[col]) for col in df.columns}, index=df.index)
    if bold:
        return_df = boldify_headers(return_df)    
    return_df.columns.names =df.columns.names
        
    return return_df


# In[Testing]


def test_SR(returns):
    """Perfomr SR-test for equal Sharpe ratios."""
    r1=returns.iloc[:, 0]
    r2=returns.iloc[:, 1]
    
    # Compute the means
    mu1 = np.mean(r1)
    mu2 = np.mean(r2)
    
    # Compute the covariance matrix
    Sigma = np.cov(r1, r2)
    
    sigma1 = np.sqrt(Sigma[0, 0])
    sigma2 = np.sqrt(Sigma[1, 1])
    sigma12 = Sigma[0, 1]
    
    # Compute theta
    theta = (2*sigma1**2*sigma2**2-2*sigma1*sigma2*sigma12
             +.5*mu1**2*sigma2**2+.5*mu2**2*sigma1**2
             -mu1*mu2/(sigma1*sigma2)*sigma12**2
             )
    
    # Compute the Jackknife statistic zjk
    test_statistic = ((sigma2 * mu1 - sigma1 * mu2)
                      / np.maximum(np.finfo(float).eps, np.sqrt(theta))
                      )
    
    p_value = norm.cdf(test_statistic)
    return p_value
    
    
def test_SD(returns):
    """Perfomr F-test for equal variances."""

    T = returns.shape[0]
    # Calculate the sample variances
    var1 = np.var(returns.iloc[:, 0], ddof=1)
    var2 = np.var(returns.iloc[:, 0], ddof=1)

    # F-statistic: ratio of variances (larger variance in the numerator)
    f_stat = var1 / var2
    
    # Degrees of freedom
    df1 = df2 = T - 1

    # Compute the one-tailed p-value (right tail, variance1 > variance2)
    p_value = 1 - f.cdf(f_stat, df1, df2)

    return p_value


def test_CE(risk_aversion, returns):
    """
    Purpose: Delta method for CE test
    
    Input:
    risk_aversion_gamma = gamma parameter (risk aversion)
    returns = Tx2 matrix of out-of-sample portfolio returns
    
    Output:
    pvalue = p-value for t-test
    
    Note: CE difference is computed for returns[:, 0] - returns[:, 1]
    """

    # Separate the returns into two portfolios
    returns_1 = returns.iloc[:, 0]
    returns_2 = returns.iloc[:, 1]

    # Derivative vector based on the CE difference and gamma parameter
    derivative_vector = np.array([1, -1, -risk_aversion/2, risk_aversion/2])

    # Number of observations (time periods)
    num_observations = len(returns_1)

    # Covariance matrix and its components
    covariance_matrix = np.cov(returns_1, returns_2)
    sigma12 = covariance_matrix[0, 1]
    sigma11 = covariance_matrix[0, 0]
    sigma22 = covariance_matrix[1, 1]

    # Standard error calculation
    covariance_structure_matrix = np.array(
        [[sigma11, sigma12, 0, 0], [sigma12, sigma22, 0, 0],
         [0, 0, 2 * sigma11**2, 2 * sigma12**2],
         [0, 0, 2 * sigma12**2, 2 * sigma22**2]])
    
    standard_error = np.sqrt((derivative_vector.T @ covariance_structure_matrix
                              @ derivative_vector) / num_observations)

    # CE difference (certainty equivalent difference)
    ce_difference = (np.mean(returns_1) - np.mean(returns_2) - 
                     (risk_aversion/2) * (np.var(returns_1)
                                          - np.var(returns_2)))

    # Test statistic and p-value
    test_statistic = ce_difference / standard_error
    p_value = norm.cdf(test_statistic)

    return p_value


def ce(x, risk_aversion=1, axis=1):
    return np.mean(x, axis=axis) - np.var(x, axis=axis)/2

def sharpe(x, axis=1):
    return np.mean(x, axis=axis)/np.std(x, axis=axis)


def test_statistic_bootstrap(returns, number_simulations=1000,
                             statistic=np.mean, block_size=10, seed=None):
    """Calculate bootstrapped p-values for differences in test statistic"""
    p = 1 / block_size
    T = returns.shape[0]
    bootstrap_samples = pd.DataFrame(index=range(number_simulations),
                                     columns=[returns.columns]
                                     )
    for i in range(number_simulations):
        starting_point = np.random.randint(0, T, block_size)
        block_length = np.random.geometric(p=p, size=block_size)
        current_bootstrap_sample = pd.DataFrame()
        curr_indices = pd.DataFrame()
        for ii in range(block_size):
            if len(current_bootstrap_sample) > block_size:
                break
            else:
                start_index = starting_point[ii]
                end_index = min(start_index + block_length[ii], T)
                index_range = pd.DataFrame(np.arange(start_index, end_index))
                current_part_sample = returns[start_index:end_index]
                current_bootstrap_sample = pd.concat(
                    [current_bootstrap_sample, current_part_sample], axis=0)
                curr_indices = pd.concat([curr_indices, index_range], axis=0)
        cutoff = len(current_bootstrap_sample) - block_size
        if cutoff > 0:
            current_bootstrap_sample = current_bootstrap_sample[:-cutoff]
            curr_indices = curr_indices[:-cutoff]
        if hasattr(statistic(current_bootstrap_sample, axis=0), "values"):
            bootstrap_samples.loc[i, returns.columns] = statistic(current_bootstrap_sample, axis=0).values
        else:
            bootstrap_samples.loc[i, returns.columns] = statistic(current_bootstrap_sample, axis=0)
        
        bootstrap_samples = bootstrap_samples.astype(float)

    # compute statistic(dataset1) - statistic(dataset2)
    results = - bootstrap_samples.diff(axis=1).iloc[:,-1]
    
    p_value = np.mean(results.values<0)

    # returnp_value
    return p_value
    

def plot_normalized_heatmap(data, normalize_by='row', annot_format=".2f", **kwargs):
    """
    Plots a heatmap with normalized colors per row or column, but annotations show the original values.
    
    Parameters:
    - data: pd.DataFrame, input data for the heatmap.
    - normalize_by: 'row' or 'column', the axis to normalize by.
    - kwargs: additional keyword arguments for seaborn.heatmap.
    """
    if normalize_by == 'row':
        norm_data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
    elif normalize_by == 'column':
        norm_data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    else:
        raise ValueError("normalize_by should be either 'row' or 'column'")
        
    # Plot heatmap with normalized colors but original annotations
    fig = plt.figure()
    plt.grid(False)
    plt.gca().xaxis.set_ticks_position('top') 
    plt.gca().xaxis.set_label_position('top')
    sns.heatmap(norm_data, linewidth=1, annot=data, fmt=annot_format, cbar=False, xticklabels=True, yticklabels=True, **kwargs)
    plt.xticks(rotation=20, ha='left')
    plt.autoscale(enable=True, axis='x', tight=True)
    # plt.show()

# test_statistic_bootstrap(returns, number_simulations=1000, statistic=np.mean, block_size=10, seed=None)
# test_statistic_bootstrap(returns, number_simulations=1000, statistic=np.std, block_size=100, seed=None)
# test_statistic_bootstrap(returns, number_simulations=1000, statistic=sharpe, block_size=10, seed=None)


# # Example return series (replace with your actual data)
# series1 = returns.iloc[:, 0]
# series2 = returns.iloc[:, 1]

# # Function to compute variance difference
# def compute_variance_difference(s1, s2):
#     return np.var(s1, ddof=1) - np.var(s2, ddof=1)

# # Stationary bootstrap resampling
# def stationary_bootstrap(series, p, size=None):
#     """ 
#     Perform stationary bootstrap.
    
#     series: Original time series
#     p: Probability of starting a new block
#     size: Length of the bootstrap sample
#     """
#     n = len(series)
#     if size is None:
#         size = n
        
#     indices = np.zeros(size, dtype=int)
#     indices[0] = np.random.randint(0, n)
    
#     for i in range(1, size):
#         if np.random.rand() < p:
#             # Start a new block
#             indices[i] = np.random.randint(0, n)
#         else:
#             # Continue the current block
#             indices[i] = (indices[i-1] + 1) % n
    
#     return series[indices]

# # Observed difference in variances
# observed_diff = compute_variance_difference(series1, series2)

# # Stationary bootstrap procedure
# def stationary_bootstrap_variance_test(s1, s2, num_bootstrap=1000, p=0.1):
#     """
#     Perform a bootstrap test for the difference in variances using stationary bootstrap.
    
#     s1, s2: Original time series
#     num_bootstrap: Number of bootstrap resamples
#     p: Probability of starting a new block in stationary bootstrap
#     """
#     n1, n2 = len(s1), len(s2)
    
#     bootstrap_diffs = []
    
#     for _ in range(num_bootstrap):
#         # Perform stationary bootstrap on both series
#         resampled1 = stationary_bootstrap(s1, p, size=n1)
#         resampled2 = stationary_bootstrap(s2, p, size=n2)
        
#         # Compute the difference in variances for the resampled data
#         bootstrap_diff = compute_variance_difference(resampled1, resampled2)
#         bootstrap_diffs.append(bootstrap_diff)
    
#     bootstrap_diffs = pd.Series(bootstrap_diffs).values
#     # Compute the p-value: proportion of bootstrap samples with a difference in variances
#     # greater than or equal to the observed difference
#     p_value = np.mean(bootstrap_diffs<0)
    
#     return observed_diff, p_value, bootstrap_diffs

# # Run the stationary bootstrap test
# observed_diff, p_value, bootstrap_diffs = stationary_bootstrap_variance_test(series1, series2)

# # Results
# print(f"Observed difference in variances: {observed_diff}")
# print(f"P-value from stationary bootstrap test: {p_value}")

# # Conclusion
# alpha = 0.05
# if p_value < alpha:
#     print("Reject the null hypothesis. The variances are significantly different.")
# else:
#     print("Fail to reject the null hypothesis. No significant difference in variances.")




