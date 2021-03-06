# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:27:04 2022

@author: renec

dependencies: numpy, panda,s scipy, yfinance

"""

import functools
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import yfinance as yf

def log_ret(x):
    return np.log(x)-np.log(x).shift(1)

def get_loss(ticker_list):
    ''' get losses (negative of returns) '''
    ret = yf.download(ticker_list,start='2019-01-01',
                                    end='2022-06-01')['Adj Close']
    ret.dropna(axis=0,inplace=True) # for dropna, rows is 0
    ret = ret.apply(func=log_ret,axis=0) # for apply, across columns is 0 -_-
    ret.dropna(axis=0,inplace=True)
    loss = -ret # negative of return for loss calculations
    return loss 
def normalize_loss(loss):
    ''' 
    Remove drift and normalize loss so units in standard deviations.
    Drift can be ignored considering exercise is to examine daily returns over
    30 day period
    '''
    stdev    = loss.std()
    normalize_loss = (loss - loss.mean())/stdev 
    return normalize_loss, stdev

def get_marginals(loss):
    ''' fix mean daily return at zero 
    and estimate marginal dists of daily losses '''
    marginal_cdfs = loss.copy()
    marginal_params = [None]*len(marginal_cdfs.columns)
    for idx,col in enumerate(marginal_cdfs):
        t_fit = stats.t.fit(marginal_cdfs[col],floc=0) # fix mean to floc=0
        marginal_params[idx] = t_fit
        marginal_cdfs[col] = stats.t.cdf(loss[col],
                                         df=t_fit[0],
                                         loc=t_fit[1],scale=t_fit[2])
    return marginal_cdfs, marginal_params

def fit_gauss_copula(marginal_cdfs):
    ''' A copula is a function that combines several marginal distributions 
    to form a joint distribution. 
    
    Gaussian Copula: C(u1,u2) = Phi(Phi^-1(u1), Phi^-1(u2))
    where u1 = F1(x1), u2 = F2(x2)
    Gaussian Copula density: c(u1,u2) = phi(phi^-1(u1),phi^-1(u2))
    
    Parametric pseudo-maximum likelihood: 
        1) Assume marginals are t-distributed; fit F_i
        2) Generate u1,u2= F1(x1),F2(x2)
        3) Estimate Rho using MLE of standard joint gaussian (sample Cov)
    
    Return: Correlation matrix that characterizes N_d(0,Corr) copula
    '''    
    # Find rho: gaussian Characterized by its mean and covariance
    args = marginal_cdfs.apply(func=stats.norm.ppf,axis=0)
    corr = args.corr()
    return corr
    
def fit_gumbel_copula(marginal_cdfs):
    ''' 
    Estimate Gumbel Copula density function using maximum-likelihood estimator.
    
    See: https://sdv.dev/Copulas/api/copulas.bivariate.gumbel.html
    
    C(u1,u2)=exp(-arg1^(1/t)), 
    
    density = partial C(u1,u2)/partial u1 u2 = 
    exp(-(arg1)^(1/t))/(arg2) * arg1^(2/t -2)/(arg3)^(1-t) 
        * (1 + (t-1)(arg1^(-1/t)))
    
    where 
        arg1= sum_i (-ln u_i)^t
        arg2= prod_i u_i
        arg3 = (prod_i ln u_i)
    '''
    
    def obj(t0):
        t = 1 + np.exp(t0) # keep theta between 1 and infinity
        arg1 = 0
        arg2 = 1
        arg3 = 1
        for col in marginal_cdfs:
            arg1 = arg1 + (-np.log(marginal_cdfs[col]))**t
            arg2 = arg2 * marginal_cdfs[col]
            arg3 = arg3 * np.log(marginal_cdfs[col])
        obj = (-arg1)**(1/t) / arg2 * arg1**(2/t-2) / arg3**(1-t)
        obj = obj * (1 + (t-1)*(arg1)**(-1/t))
        obj = np.log(obj)
        obj = obj.sum()
        return -obj
    t0 = 1.5
    fit = optimize.minimize(obj,x0=t0, method='nelder-mead')
    t = 1 + np.exp(fit.x)
    return t

def sim_gauss_copula(corr,size):
    ''' 
    Take in correlation matrix -- corr -- as argument for Gauss copula 
    '''
    U = np.random.multivariate_normal(mean=[0,0],cov=corr,size=size)
    U = np.apply_along_axis(stats.norm.cdf, 1, U)
    U = np.split(U,indices_or_sections=2,axis=1)
    for i in range(len(U)):
        U[i] = U[i].reshape(1,size)[0] 
    return U

def sim_gumbel_copula(t,size):
    '''
    Take in scalar -- t -- as argument for Gumbel copula 
    
    Algorithm: 
        1) Generate independent uniform random variables: u, p
        2) Solve for v that satisfies the conditional distribution
            p = Prob(V<v|U=u), when u,v ~ C(u,v) Gumbel Copula
    '''

    u = np.random.uniform(low=0,high=1,size=size)
    p = np.random.uniform(low=0,high=1,size=size)
    def cond_cdf(v,u,t,p):
        ''' cdf of v conditional on u for bivariate copula given param t '''
        cond_cdf = np.exp(-((-np.log(u))**t+(-np.log(v))**t)**(1/t))
        cond_cdf = cond_cdf *((-np.log(u))**t+(-np.log(v))**t)**(1/t-1)
        cond_cdf = cond_cdf *(-np.log(u))**(t-1)/u - p
        return cond_cdf
    
    cond_cdf_to_solve = functools.partial(cond_cdf,u=u,t=t,p=p)
    
    v0 = np.random.uniform(low=0,high=1,size=size)
    lb = [0]*size
    ub = [1]*size
    # solve and restrict search to (0,1) interval
    v_solution = optimize.least_squares(cond_cdf_to_solve,x0=v0,bounds=(lb,ub))
    U = [u,v_solution.x]
    return U

def sim_losses(copula_params,marginal_params,stdev,S=10**3,size=30):
    corr, t = copula_params
    def sim_cumloss(sim):
        for i in range(len(sim)):
            sim[i] = stdev[i]*stats.t.ppf(sim[i],df = marginal_params[i][0],
                                          loc = 0,
                                          scale = marginal_params[i][2])
            sim[i] = np.sum(sim[i])
        return 0.5 * sim[0] + 0.5* sim[1]
    
    portfolio_gauss = [None]*S
    for s in range(S):
        sim = sim_gauss_copula(corr,size)
        portfolio_gauss[s] = sim_cumloss(sim)
        
    portfolio_gumbel = [None]*S
    for s in range(S):
        sim = sim_gumbel_copula(t,size)
        portfolio_gumbel[s] = sim_cumloss(sim)
        
    return portfolio_gauss, portfolio_gumbel

def main():
    # Get Stock Data
    ticker_list = ['AAPL','XOM']
    loss = get_loss(ticker_list) 
    
    # Fit marginal distributions
    loss_norm, stdev = normalize_loss(loss)
    marginal_cdfs, marginal_params = get_marginals(loss_norm)
    
    # Fit copulae
    corr = fit_gauss_copula(marginal_cdfs)
    t    = fit_gumbel_copula(marginal_cdfs)
    copula_params = [corr,t]
    
    # Simulate returns from copulae
    np.random.seed(seed=18) # set seed to reproduce results
    portfolio_gauss, portfolio_gumbel = sim_losses(copula_params,
                                                   marginal_params,
                                                   stdev,
                                                   S=1000,size=30)
    
    # Historical returns, 30 rolling period, baseline
    loss['portfolio'] = 0.5 * loss['AAPL'] + 0.5 * loss['XOM'] 
    # 30day var
    portfolio_30d = loss['portfolio'].rolling(30).sum()
    portfolio_30d.dropna(axis=0,inplace=True)
    
    # percentiles of interest
    p = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
    
    print('Historical, rolling 30 days')
    print(portfolio_30d.describe(p))
    print('')
    
    print('Model, Gaussian Copula, t-dist marginals')
    print(pd.DataFrame(portfolio_gauss,columns=['Gauss']).describe(p))
    print('')
    
    print('Model, Gumbel Copula, t-dist marginals')
    print(pd.DataFrame(portfolio_gumbel,columns=['Gumbel']).describe(p))
    print('')
    
    print('VaR - alpha = 0.05')
    VaR = {'Historical':np.quantile(portfolio_30d,0.95).round(4),
           'Gaussian':np.quantile(portfolio_gauss,0.95).round(4),
           'Gumbel':np.quantile(portfolio_gumbel,0.95).round(4)}
    print(VaR)
    
if __name__ == "__main__":
    main()