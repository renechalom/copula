# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:27:04 2022

@author: renec

dependencies: yfinance

"""

import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def log_ret(x):
    return np.log(x)-np.log(x).shift(1)

def get_ret(ticker_list):
    ret = yf.download(ticker_list,start='2019-01-01',
                                    end='2022-06-01')['Adj Close']
    ret.dropna(axis=0,inplace=True) # for dropna, rows is 0
    ret = ret.apply(func=log_ret,axis=0) # for apply, across columns is 0 -_-
    ret.dropna(axis=0,inplace=True)
    return ret

def fit_gauss_copula(ret):
    ''' A copula is a function that combines several marginal distributions 
    to form a joint distribution. 
    
    Gaussian Copula: C(u1,u2) = Phi(Phi^-1(u1), Phi^-1(u2))
    where u1 = F1(x1), u2 = F2(x2)
    Gaussian Copula density: c(u1,u2) = phi(phi^-1(u1),phi^-1(u2))
    
    Parametric pseudo-maximum likelihood: 
        1) Assume marginals are t-distributed; fit F_i
        2) Generate u1,u2= F1(x1),F2(x2)
        3) Estimate Rho using MLE of standard joint gaussian
    '''
    marginal_cdfs = ret.copy()
    for col in marginal_cdfs:
        t_fit = stats.t.fit(marginal_cdfs[col],floc=0) # fix mean to floc=0
        marginal_cdfs[col] = stats.t.cdf(ret[col],
                                         df=t_fit[0],
                                         loc=t_fit[1],scale=t_fit[2])
    
    # Find rho: gaussian Characterized by its mean and covariance
    args = marginal_cdfs.apply(func=stats.norm.ppf,axis=0)
    corr = args.corr()
    return corr
    
def main():
    ticker_list = ['AAPL','XOM']
    ret = get_ret(ticker_list)

if __name__ == "__main__":
    main()