# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:31:14 2021

@author: Frey
"""

import pandas as pd
import numpy as np


def mdd(data):
    ''' mdd: Returns maximum drawdown for return series
    '''
    try:
        i = (np.maximum.accumulate(data)-data).values.argmax()
        j = (data[:i]).values.argmax()
        mdd0 = data.iloc[j]-data.iloc[i]
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