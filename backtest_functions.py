# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:12:33 2021

@author: frey
"""
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cvxpy
from aggregated_functions import (shift_intervall, annual_measures,
                                  total_measures, turnover_measures,
                                  hierarchical_ridge, bayesian_lasso,
                                  bayesian_elastic_net, truncted_normal,
                                  ledoit_wolf, frahm_memmel, tou_zhou,
                                  fama_french, test_SD, test_SR, test_CE,
                                  test_statistic_bootstrap, sharpe, ce,
                                  plot_normalized_heatmap)
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
import math
import pandas_datareader
import os
import pickle
import seaborn as sns


def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) - dt.timedelta(days=1)


# from sklearn.linear_model import Lasso
# weights_tilde = Lasso(alpha=100).fit(X, y).coef_

# y = np.array(returns.mean(axis=1))
# x = np.array(-returns.subtract(y, axis=0))
# x = np.insert(x,0,1,axis=1)

class PortfolioBacktest(object):
    '''
    PortfolioBacktest: Calculate backtest for strategies
    '''

    def __init__(self):
        self.data = dict()
        self.settings = dict()

    def get_data(self):
        '''
        get_data: clean and adjust chosen data
        '''

        # return data
        file_name = os.path.join(self.settings['data_path'],
                                 f"{self.settings['data_set']}.pkl")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as file:
                return_data, factor_data, check_data = pickle.load(file)
        else:
            start_date_data = '19630701'
            ff_datasets = pandas_datareader.famafrench.get_available_datasets()
            if self.settings['data_set'] in ff_datasets:
                return_data = (pandas_datareader.famafrench.FamaFrenchReader(
                    self.settings['data_set'], start=start_date_data, end=None)
                    .read()
                    )
                return_data = return_data[0].copy()
                return_data.index = return_data.index.to_timestamp()
                return_data = return_data.reset_index()
                return_data['Date'] = return_data['Date'].apply(last_day_of_month)
                return_data = return_data.set_index('Date')

                # clean data
                return_data[return_data<=-99.99] = 0 # cut NaNs
                return_data = return_data/100 # decimal returns

                check_data = pd.DataFrame(True, index=return_data.index, columns=return_data.columns)

            else:
                number_assets = int(self.settings['data_set'].split('_')[0])
                dropbox_link = 'https://www.dropbox.com/scl/fi/fmrtpkjzk8dzz10xspg26/DatastreamData.xlsx?rlkey=2qx99p4r9kwzlddonebsx8hh0&dl=1'
                price_data = pd.read_excel(
                    dropbox_link,
                    sheet_name=f"{self.settings['data_set'].split('_')[-1]}_MV_RI",
                    engine='openpyxl', index_col=0
                    )
                # tidy dataframe
                melted_data = (price_data
                               .reset_index(drop=False).melt(
                                   id_vars='Name',
                                   var_name='company_type',
                                   value_name='value')
                               .assign(company=lambda x: x['company_type'].str.split(' - ', expand=True)[0],
                                       type=lambda x: x['company_type'].str.split(' - ', expand=True)[1])
                               .assign(type=lambda x: x['type']
                                      .str.replace('MARKET VALUE', 'market_value', case=False)
                                      .str.replace('TOT RETURN IND', 'total_return', case=False))
                               .drop(columns=['company_type'])
                               .rename(columns={'Name': 'date'})
                               .get(['date', 'company', 'type', 'value'])
                               # .assign(market_rank=lambda df: df[df['type'] == 'market_value']
                               #         .groupby('date')['value']
                               #         .rank(ascending=False)
                               #         )
                               )

                return_data = (melted_data
                               .query('type == "total_return"')
                               .pivot(index='date', columns='company', values='value')
                               .pct_change(1)
                               )
                return_data = return_data[return_data.isna().sum(axis=1) != return_data.shape[1]]
                return_data.index = pd.to_datetime(return_data.index)
                return_data.index = return_data.index.where(return_data.index.is_month_end, return_data.index + pd.offsets.MonthEnd(0))
                return_data.index.name = 'Date'
                return_data.columns.name = None
                mv_data = (melted_data
                           .query('type == "market_value"')
                           .pivot(index='date', columns='company', values='value')
                           # .rank(axis=1, ascending=False)
                            )
                mv_data = mv_data[mv_data.isna().sum(axis=1) != mv_data.shape[1]]
                mv_data.index = pd.to_datetime(mv_data.index)
                mv_data.index = mv_data.index.where(mv_data.index.is_month_end, mv_data.index + pd.offsets.MonthEnd(0))
                return_data.index.name = 'Date'
                return_data.columns.name = None

                # Check assets availability
                check_data_backward = return_data.rolling(self.settings['window']+1, closed='left').count() == self.settings['window']+1
                check_data_forward = return_data.shift(-self.settings['forward_window']-1).rolling(self.settings['forward_window'], closed='left').count() == self.settings['forward_window']
                check_availability = check_data_backward * check_data_forward

                check_availability = check_availability.reindex(mv_data.index).fillna(False)

                # Function to rank values based on the mask
                def rank_row(values, mask_row):
                    # Get only the values where the mask is True
                    filtered_values = values[mask_row]
                    # Rank only the filtered values
                    ranked = pd.Series(filtered_values).rank(method='first', ascending=False).values
                    # Create an array of NaNs, and fill with ranks in the positions of True in the mask
                    result = np.full(values.shape, np.nan)
                    result[mask_row] = ranked
                    return result

                # Apply the ranking function row-wise without looping explicitly
                mv_rank_data = mv_data.apply(lambda row: rank_row(row.values, check_availability.loc[row.name].values), axis=1)

                # Convert back to a numpy array if needed
                mv_rank_data = pd.DataFrame(np.array(mv_rank_data.tolist()).tolist(), columns=mv_data.columns, index=mv_data.index)

                # Check correlations and market value
                all_correlations=return_data.rolling(self.settings['window']).corr()
                check_data_correlation = (all_correlations >= self.settings['correlation_threshold']) & (all_correlations < 0.99)
                correlation_indices = check_data_correlation.stack()[check_data_correlation.stack() == True].index
                correlation_indices.names = ['date', 'company1', 'company2']

                # check something
                # correlation_indices_test = pd.DataFrame(index=correlation_indices).reset_index(drop=False)
                # correlation_indices_test['rank1'] = 0
                # correlation_indices_test['rank2'] = 0
                # import datetime
                # for i in correlation_indices_test.index:
                #     tmp_date = datetime.datetime.strftime(correlation_indices_test.loc[i, 'date'], '%Y-%m-%d')
                #     tmp_comp1 = correlation_indices_test.loc[i, 'company1']
                #     tmp_comp2 = correlation_indices_test.loc[i, 'company2']
                #     correlation_indices_test.loc[i, 'rank1'] = mv_rank_data.loc[tmp_date, tmp_comp1]
                #     correlation_indices_test.loc[i, 'rank2'] = mv_rank_data.loc[tmp_date, tmp_comp2]
                
                check_data_mv_correlation = (mv_rank_data <= number_assets).copy()

                last_date = correlation_indices[0][0]
                tmp_number_assets = number_assets
                max_number_available_assets = check_data_mv_correlation.astype(int).sum(axis=1)
                for i in correlation_indices:
                    if i[0] > last_date:
                        tmp_number_assets = number_assets

                    if mv_rank_data.loc[i[0], [i[1], i[2]]].max() <= tmp_number_assets:
                        tmp_replace_asset = mv_rank_data.loc[i[0], [i[1], i[2]]].idxmax()
                        if check_data_mv_correlation.loc[i[0], tmp_replace_asset] == True:
                            check_data_mv_correlation.loc[i[0], tmp_replace_asset] = False
                            tmp_number_assets += 1
                            
                            # replace asset
                            new_asset = mv_rank_data.loc[i[0]] == tmp_number_assets
                            if len(new_asset[new_asset]) == 1:
                                check_data_mv_correlation.loc[i[0], new_asset[new_asset].index[0]] = True

                    last_date = i[0]
                
                check_data = check_availability * check_data_mv_correlation
            
            factor_data = pandas_datareader.famafrench.FamaFrenchReader("F-F_Research_Data_5_Factors_2x3",
                                                                    start=start_date_data, end=None).read()
            factor_data = factor_data[0].copy()
            factor_data.index = factor_data.index.to_timestamp()
            factor_data = factor_data.reset_index()
            factor_data['Date'] = factor_data['Date'].apply(last_day_of_month)
            factor_data = factor_data.set_index('Date')
            factor_data = factor_data[['Mkt-RF', "SMB", "HML", "RMW", "CMA"]]
            
            # clean data
            factor_data[factor_data<=-99.99] = 0 # cut NaNs
            factor_data = factor_data/100 # decimal returns
            
            factor_data.index = factor_data.index.where(factor_data.index.is_month_end, factor_data.index + pd.offsets.MonthEnd(0))
            
            factor_data = factor_data.reindex(return_data.index)
            
            with open(file_name, 'wb') as file:
                pickle.dump([return_data, factor_data, check_data], file)

        # save data
        self.data['returns'] = return_data
        self.data['factor_returns'] = factor_data
        self.data['check_data'] = check_data
        self.data['index'] = pd.DataFrame(return_data.mean(axis=1),columns=['Index'])
        self.data['dates'] = return_data.index
        return
    def calc_allocation(self, returns_all, opt_method, factors):
        ''' calc_allocation: Specify minimization algorithm and calculate weights
        - initial: initial weight guess
        '''
        # Check if data is complete (only use columns where there is actual data)
        returns = returns_all.loc[:, (returns_all.sum(axis=0) !=0)].copy()
        number_simulations = self.settings['number_simulations']

        if opt_method == '1/N': # 1/N portfolio

            strategy_name = opt_method
            final_weights = np.ones((returns.shape[1], 1))/returns.shape[1]

        if opt_method == 'GMVP': # GMVP (no constraints)

            strategy_name = opt_method
            y = np.array(returns.iloc[:, 0])
            x = np.array(-returns.iloc[:,1:].subtract(y, axis=0))
            x = np.insert(x,0,1,axis=1)
            weights = np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
            final_weights = np.append(1-weights.sum(),weights[1:]).round(8)

        if opt_method == 'Ridge':  # GMVP with Ridge tau=T/N

            strategy_name = opt_method
            T, N = returns.shape
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)
            tau = T/N
            weights_tilde = np.linalg.pinv(np.eye(N+1)/tau+X.T.dot(X)).dot(X.T.dot(y))
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        if opt_method == 'EmpBayes':  # Empirical Bayes

            strategy_name = 'Empirical Bayes'
            T, N = returns.shape
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)
            tau = T/N/100 #1/(T*N/100)#T/N
            # if X.shape[0] > X.shape[1]:
            #     weights_tilde = (tau/(tau+1)) * np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
            # else:
            #     weights_tilde = np.full((X.shape[1],), np.nan)
            weights_tilde = (tau/(tau+1)) * np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 'HierRidge':  # Hierarchical Ridge

            strategy_name = 'Hierarchical Ridge'
            N = returns.shape[1]
            w_0 = np.ones([N, 1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y, N, axis=1) - np.array(returns)
            X = np.insert(X, 0, 1, axis=1)

            beta, sigma, tau = hierarchical_ridge(y, X, number_simulations)
            weights_tilde = beta.mean(axis=0)
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 'Lasso':  # Lasso

            strategy_name = 'Lasso'
            N = returns.shape[1]
            w_0 = np.ones([N, 1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y, N, axis=1) - np.array(returns)
            X = np.insert(X, 0, 1, axis=1)

            weights_tilde = LassoCV().fit(X, y).coef_

            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 'BayLasso':  # Bayesian Lasso

            strategy_name = 'Bayesian Lasso'
            N = returns.shape[1]
            w_0 = np.ones([N, 1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y, N, axis=1) - np.array(returns)
            X = np.insert(X, 0, 1, axis=1)

            beta, sigma, invtau2, lambda_out = bayesian_lasso(y, X, number_simulations)
            weights_tilde = beta.mean(axis=0)

            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 'ElasticNet':  # Elastic Net

            strategy_name = 'Elastic Net'
            N = returns.shape[1]
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)
            
            weights_tilde = ElasticNetCV().fit(X, y).coef_
            
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())
            
        elif opt_method == 'BayElasticNet':  # Bayesian Elastic Net

            strategy_name = 'Bayesian Elastic Net'
            N = returns.shape[1]
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)

            beta, sigma, invtau2, lambda1_out, lambda2_out = bayesian_elastic_net(y, X, number_simulations)
            weights_tilde = beta.mean(axis=0)
                        
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 'Truncted Normal': # Truncted Normal
        
            ###test for now
            strategy_name = opt_method
            y = np.array(returns.iloc[:, 0])
            x = np.array(-returns.iloc[:,1:].subtract(y, axis=0))
            x = np.insert(x,0,1,axis=1)
            weights = np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
            final_weights = np.append(1-weights.sum(),weights[1:]).round(8)
            
            final_weights = ((final_weights - final_weights.min())
                             / (final_weights.max() - final_weights.min())
                             )
        

            # strategy_name = opt_method
            # N = returns.shape[1]
            # w_0 = np.ones([N,1]) / N
            # y = np.matmul(np.array(returns), w_0)
            # X = np.repeat(y,N,axis=1) - np.array(returns)
            # X = np.insert(X,0,1,axis=1)

            # beta, sigma = truncted_normal(y, X, number_simulations, -1/N, 1/N)
            # weights_tilde = beta.mean(axis=0)
            # final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 'LW':  # Ledoit & Wolf

            strategy_name = 'LW'
            weights, Sigma = ledoit_wolf(returns, 2)
            final_weights = weights

        elif opt_method == 'FM':  # Frahm & Memmel

            strategy_name = 'FM'
            weights = frahm_memmel(returns, 2)
            final_weights = weights

        elif opt_method == 'TZ':  # Tou & Zhou

            strategy_name = 'TZ'
            weights = tou_zhou(returns, 1)
            final_weights = weights

        elif opt_method == 'FF': # Fama French

            strategy_name = 'FF'
            weights = fama_french(returns, factors)
            final_weights = weights

        elif opt_method == '1/vol': # 1/vol portfolio

            strategy_name = '1/vol'
            final_weights = np.array(1/np.std(returns))
            final_weights = final_weights / final_weights.sum()

        elif opt_method == 'GMVP+': # GMVP (no short sales)

            strategy_name = 'GMVP+'
            out_weights = cvxpy.Variable((returns.shape[1], 1))
            out_weights.value  = np.ones((returns.shape[1], 1))/returns.shape[1]
            lincon = [out_weights >= self.settings['lower'],\
                      out_weights <= self.settings['upper'],\
                      sum(out_weights) == 1]
            Sigma = np.cov(returns.T)
            obj = cvxpy.Minimize(cvxpy.quad_form(out_weights, Sigma))
            prob = cvxpy.Problem(obj, lincon)
            prob.solve(solver=cvxpy.SCS, warm_start=False, verbose=False, use_indirect=True)
            final_weights = out_weights.value

        elif opt_method == 'ERC': # equal risk contribution portfolio

            strategy_name = 'ERC'
            from scipy.optimize import minimize
            TOLERANCE = 1e-20
            initial_weights = np.ones((returns.shape[1], 1))/returns.shape[1]
            Sigma = np.cov(returns.T)
            assets_risk_budget = [1 / returns.shape[1]] * returns.shape[1]

            def risk_parity_objective(weights, args):

                # arguments for optimization
                covariances = args[0]
                assets_risk_budget = args[1]

                # transformation
                weights = np.matrix(weights)

                # portfolio volatility
                portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

                # risk contribution and target of each asset
                risk_contributions = np.multiply(weights.T, covariances * weights.T) / portfolio_risk
                risk_targets = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))
                risk_differences = sum(np.square(risk_contributions - risk_targets.T))[0, 0]

                return risk_differences

            # Long constraints and sum equals 100%
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                           {'type': 'ineq', 'fun': lambda x: x})

            # Optimisation process in scipy
            optimize_result = minimize(fun=risk_parity_objective,
                                       x0=initial_weights,
                                       args=[Sigma, assets_risk_budget],
                                       method='SLSQP',
                                       constraints=constraints,
                                       tol=TOLERANCE,
                                       options={'disp': False})

            final_weights = optimize_result.x
        # Process results
        if final_weights.ndim == 1:
            final_weights = final_weights[:, np.newaxis]
        allocation = pd.DataFrame(final_weights)
        allocation.index = returns.columns
        # allocation = allocation.groupby(by=allocation.index).sum(numeric_only=False)skipna=True)
        # allocation = allocation.fillna(0)
        allocation.index.name = strategy_name
        allocation.columns = ['weights']
        return allocation
    def rebalancing_dates(self):
        '''
        rebalancing_dates: calculte rebalancing dates for given vector of dates in time series
        Input:
        - dates: list of dates, e.g. RETURNS.index
        - period: string, e.g. days, weeks, months, years
        - frequency: scalar, gives observations frequency, e.g. every day, every second day etc.
        - method:  scalar = 1 for specifying which element in period, e.g. first, second,... or last etc.
                          = 2 for specifying which day of the week, Monday = 0, Friday = 4.
        - day: scalar
        '''
        period = self.settings['rebalancing_period']
        frequency = self.settings['rebalancing_frequency']
        method = 1
        day = 0
        position = 0
        start = self.settings['start_date']
        end = self.settings['end_date']
        all_dates = pd.DataFrame(self.data['returns'].loc[start:end].index).rename(columns={'Date': 'date', 0: 'date'})

        # first rebalancing period (days, week, months, years)
        if period == 'days':
            all_dates['period'] = all_dates['date'].dt.strftime('%Y-%d')
        elif period == 'weeks':
            all_dates['period'] = all_dates['date'].dt.strftime('%Y-%V')
        elif period == 'months':
            all_dates['period'] = all_dates['date'].dt.strftime('%Y-%m')
        elif period == 'years':
            all_dates['period'] = all_dates['date'].dt.strftime('%Y')

        # add weekdays and drop weekend
        all_dates['dayofweek'] = all_dates['date'].dt.dayofweek
        # all_dates.drop(all_dates[all_dates['dayofweek'] >= 5].index , inplace=True) # Drop weekends
        all_dates = all_dates.reset_index()

        # rebalancing frequency and period
        if method == 1:
            all_dates = all_dates.groupby('period').filter(lambda x: x.shape[0] >= abs(day)+1)
            rebalancing_dates = all_dates.groupby('period').apply(lambda x: (x.iloc[day]))
        elif method == 2:
            if position == 0:
                idx = all_dates.groupby('period').apply(lambda x: x['dayofweek'].sub(day).abs().idxmin())
                rebalancing_dates = all_dates.iloc[idx.values,:]
            else:
                idx = all_dates.groupby(['period','dayofweek']).apply(lambda x: x['dayofweek'].sub(day).abs())
                df = idx[:,day].index.to_frame().rename(columns={1: 'day'}).reset_index(drop=True)
                df['count'] = (df.groupby('period').apply(lambda x: x.reset_index()).index.get_level_values(1)).tolist()
                df = df.pivot(index='count', columns='period', values='day').T
                rebalancing_dates = all_dates.iloc[df.loc[:,position].dropna().astype(int).tolist()]
        rebalancing_dates = sorted(rebalancing_dates[::frequency]['date'].tolist())
        ########################
        ########################
        ###### For now
        rebalancing_date = sorted(self.data['returns'].index[1:].tolist())
        ########################
        ########################
        return rebalancing_dates
    def backtest_allocation(self, opt_method):
        ''' backtest_allocation: Calculate backtest
        '''
        window = self.settings['window']
        historical_allocation = []
        dates_reb = self.rebalancing_dates()
        # First step
        allocation_dates = []
        returns_all = self.data['returns']
        factor_returns = self.data['factor_returns']
        check_data = self.data['check_data']
        returns_all_ni = returns_all.copy().reset_index()
        # for i in enumerate(dates_reb):
        #     print(i)

        for i in range(0, len(dates_reb), 1):
            # First get the timing right
            tmp_date = str(dates_reb[i]).split()[0]
            
            if tmp_date == '1995-08-31':
                fgd=1
            end_window = returns_all_ni[returns_all_ni['Date'] == tmp_date].index.tolist()
            end_window = end_window[0] - 1
            start_window = end_window - window
            valid_assets = check_data.loc[tmp_date, (check_data.loc[tmp_date,:] == True).values].index.tolist()
            try:
                tmp_returns = returns_all.iloc[start_window:end_window, :][valid_assets]
                tmp_factor_returns = factor_returns.iloc[start_window:end_window, :]
            except (RuntimeError, ValueError, TypeError, NameError, IndexError):
                tmp_returns = returns_all.iloc[:end_window, :][valid_assets]
                tmp_factor_returns = factor_returns.iloc[:end_window, :]
            allocation_dates.append(tmp_date)
            
            if tmp_returns.index[-1] == pd.to_datetime('2019-06-30'):
                print("test")
            
            hist_alloc_tmp = self.calc_allocation(tmp_returns, opt_method, tmp_factor_returns)
            historical_allocation.append(hist_alloc_tmp)
        back_alloc = pd.concat(historical_allocation[0:], keys=allocation_dates[0:])
        back_dates = allocation_dates[0:]
        return back_alloc, back_dates
    def complete_allocation(self, opt_method):
        ''' complete_allocation: Returns the allocation for each day not only for allocation days
        '''
        ret = self.data['returns'][self.settings['start_date']:self.settings['end_date']]
        allocations, dates = self.backtest_allocation(opt_method)
        all_alloc = allocations['weights'].unstack()#.replace(np.nan, 0)
        all_alloc.index = pd.to_datetime(all_alloc.index)
        all_alloc = all_alloc.reindex(ret.loc[self.settings['start_date']\
                                              :self.settings['end_date']].index)#.ffill()
        for i in range(1, len(dates)):
            cum_ret = (1+ret.loc[dates[i-1]:dates[i], all_alloc.columns]).cumprod()
            cum_ret = cum_ret.iloc[:-1, :]
            all_alloc.loc[cum_ret.index, :] = all_alloc.loc[cum_ret.index, :]*cum_ret
        # cumulative performance after last rebalancing to end of the sample
        cum_ret = (1+ret.loc[dates[-1]:, all_alloc.columns]).cumprod()
        cum_ret = cum_ret.iloc[1:-1, :]
        all_alloc.loc[cum_ret.index, :] = all_alloc.loc[cum_ret.index, :]*cum_ret
        if self.settings['long_only_portfolio_weights']:
            all_alloc = pd.DataFrame([shift_intervall(all_alloc.loc[i[1],:], 0, 1) for i in enumerate(all_alloc.index)])
        all_alloc = all_alloc.div(all_alloc.sum(axis=1),axis=0)
        return all_alloc, dates
    def backtest(self):
        ''' backtest: Returns the historical performance of the allocation
        '''
        # Data
        self.get_data()
        all_returns = self.data['returns']
        window = self.settings['window']
        length_year = self.settings['length_year']
        risk_aversion = self.settings['risk_aversion']
        N = all_returns.shape[1]

        # Allocation
        allocations = dict()
        returns = dict()
        performances = dict()
        annuals = dict()
        totals = pd.DataFrame()
        turnover = pd.DataFrame()
        mean_MAD = pd.DataFrame()
        performances['returns'] = pd.DataFrame()
        performances['WD'] = pd.DataFrame()


        for i in enumerate(self.settings['opt_method']):

            tmp_allocations, dates = self.complete_allocation(i[1])
            strategy_name = tmp_allocations.columns.name
            allocations[strategy_name] = tmp_allocations
            returns[strategy_name] = all_returns[list(allocations[strategy_name].columns)].loc[self.settings['start_date']:self.settings['end_date'], :]
            weight_change = abs(allocations[strategy_name]-allocations[strategy_name].shift()).sum(axis=1)
            costs = weight_change * self.settings['costs']
            skip_na = False
            if allocations[strategy_name].isna().sum(axis=1).max() < allocations[strategy_name].shape[1]:
                skip_na = True
            performances['returns'].loc[:,strategy_name] = (allocations[strategy_name].shift() * returns[strategy_name]).sum(axis=1, skipna=skip_na) - costs
            performances['WD'].loc[:,strategy_name]  = abs(allocations[strategy_name]-1/N).sum(axis=1, skipna=skip_na)
            mean_MAD.loc['MAD', strategy_name] = abs(allocations[strategy_name]-1/N).sum(axis=1, skipna=skip_na).mean(axis=0)
            annuals[strategy_name] = annual_measures(performances['returns'].loc[:,strategy_name])
            turnover.loc[:, strategy_name], annual_turnover_tmp = turnover_measures(allocations[strategy_name], dates)
            annuals[strategy_name].loc[:, 'Turnover'] = annual_turnover_tmp[0]
            annuals[strategy_name] = round(annuals[strategy_name].astype(float),self.settings['round_decimals'])

            print('Allocation for ' + strategy_name + ' is complete.')

        if self.settings['normalized_returns']:
            performances['returns'] *= (performances['returns']['1/N'].std()
                                        / performances['returns'].std()
                                        )

        performances['AR'] = performances['returns'].rolling(window=window, min_periods=1).mean() * 100 * length_year
        performances['SD'] = performances['returns'].rolling(window=window, min_periods=1).std() * 100 * np.sqrt(length_year)
        performances['SR'] = performances['AR'].div(performances['SD'])
        performances['CE'] = performances['AR'] - (risk_aversion/2) * performances['SD']
        performances['RL'] = performances['SD'].mul(performances['AR'].loc[:,'1/N'].div(performances['SD'].loc[:,'1/N']),axis=0) - performances['AR']
        performances['DD'] = (1+performances['returns']).cumprod(axis=0)/(1+performances['returns']).cumprod(axis=0).cummax(axis=0)-1

        totals = total_measures(performances['returns'], length_year, risk_aversion)
        totals.loc['MAD', :] = mean_MAD.values
        totals = round(totals.astype(float),self.settings['round_decimals'])
        
        
        # Do some testing
        p_values = pd.DataFrame(1, index=totals.index, columns=totals.columns)
        number_simulations = 1000
        for i in performances['returns'].columns[1:]:
            
            if totals.loc[:, i].sum() == 0:
                totals.loc[:, i] *= np.nan
            
            tmp_returns = performances['returns'][['1/N', i]]
            if tmp_returns.loc[:, i].sum() == 0:
                continue
            else:
                if self.settings['p_values_bootstrapped']:
                    # SR
                    p_values.loc['Sharpe', i] = test_statistic_bootstrap(tmp_returns, number_simulations=number_simulations,
                                                                         statistic=sharpe, block_size=12, seed=None)
                    # SD
                    p_values.loc['Volatility', i] = test_statistic_bootstrap(tmp_returns, number_simulations=number_simulations,
                                                                         statistic=np.std, block_size=12, seed=None)
                    # CE
                    p_values.loc['Certainty Equivalent', i] = test_statistic_bootstrap(tmp_returns, number_simulations=number_simulations,
                                                                                       statistic=ce, block_size=12, seed=None)
                else:
                    # SR
                    p_values.loc['Sharpe', i] = test_SR(tmp_returns)
                    # SD
                    p_values.loc['Volatility', i] = test_SD(tmp_returns)
                    # CE
                    p_values.loc['Certainty Equivalent', i] = test_CE(risk_aversion, tmp_returns)
        

        # put in structure to be saved
        backtest_output = dict()
        backtest_output['performance'] = performances
        backtest_output['p_values'] = p_values
        backtest_output['annuals'] = annuals
        backtest_output['totals'] = totals
        backtest_output['allocations'] = allocations
        backtest_output['turnover'] = turnover
        backtest_output['rebalancing_dates'] = dates
        backtest_output['data'] = self.data
        backtest_output['settings'] = self.settings
        
        self.backtest_output = backtest_output
        
        # plot
        if (self.settings['plot']) and (self.settings['start_date'] != self.settings['end_date']):
            self.plot_backtest(backtest_output)

        else:
            pass

        # save structure
        years_start = str(dt.datetime.strptime(self.settings['start_date'],'%Y%m%d').year)
        years_end = str(dt.datetime.strptime(self.settings['end_date'],'%Y%m%d').year)
        pickle_file_name = os.path.join(self.settings['results_data_path'], self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] + '.pkl')
        F = open(pickle_file_name, 'wb')
        pickle.dump(backtest_output, F)
        F.close()
        
        # Write annuals and totals to Excel
        
        totals_name = os.path.join(self.settings['results_data_path'], self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] + '_totals.xlsx')
        annuals_name = os.path.join(self.settings['results_data_path'], self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] + '_annuals.xlsx')

        backtest_output['totals'].to_excel(totals_name)
        with pd.ExcelWriter(annuals_name) as writer: 
            for i in enumerate(backtest_output['allocations'].keys()):
                backtest_output['annuals'][i[1]].to_excel(writer, sheet_name=i[1].replace('/',''))

        return backtest_output
    def plot_backtest(self, backtest_output, *years):
        ''' plot_backtest: Plot backtest figures
        '''
        plot_style_type = self.settings['plot_style_type']
        colormap = 'tab20c'
        if len(years) == 0:
            years = str(dt.datetime.strptime(self.settings['start_date'],'%Y%m%d').year)
        else:
            years = years[0]
        try:
            years_end = dt.datetime.strftime(self.settings['end_date'],'%Y')
        except:
            years_end = str(dt.datetime.strptime(self.settings['end_date'],'%Y%m%d').year)
        # Data
        returns = backtest_output['performance']['returns']
        # Totals
        plot_normalized_heatmap(backtest_output['totals'].iloc[1:,:].T, normalize_by='column', cmap="RdBu", alpha=0.7)

        plt.gcf().savefig(self.settings['results_plot_path']+'\\' +self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] +'_totals' + plot_style_type,\
             dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        
        # Equity Lines
        plt.set_cmap('tab20c')
        plt.figure()
        plt.plot((1+returns).cumprod(axis=0)*100)
        plt.legend(returns.columns, frameon=False)
        # plt.yscale('log')
        plt.title(f"{self.settings['data_set']} - Performance")
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_plot_path']+'\\' +self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] +'_perf' + plot_style_type,\
             dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Equity Lines for each year
        if self.settings['plot_performance_years']:
            for i in enumerate(returns.index.year.unique()):
                plt.figure()
                try:
                    data = returns[str(i[1])]
                    data.loc[dt.datetime.strptime(str(data.index.year[0])+'0101','%Y%m%d')] = 0
                    data = data.sort_index()
                    tmp_portfolio_returns = data
                except:
                    tmp_portfolio_returns = returns[str(i[1])]
                    pass
                plt.plot(((1+tmp_portfolio_returns).cumprod()*100))
                plt.legend(returns.columns, frameon=False)
                plt.title(f"{self.settings['data_set']} - Performance in {i[1]}")
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.gcf().savefig(self.settings['results_plot_path']+'\\' +self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] +'_perf_' + str(i[1]) + plot_style_type,\
                     dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Rolling Measures
        if self.settings['plot_rolling']:
            for i in enumerate(backtest_output['performance']):
                plt.figure()
                plt.plot(backtest_output['performance'][i[1]])
                plt.legend(backtest_output['performance'][i[1]].columns, frameon=False)
                plt.title(f"{self.settings['data_set']} - Annualized Rolling {i[1]}")
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.gcf().savefig(self.settings['results_plot_path']+'\\' +self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] +'_roll_'+ i[1] + '_' + years + '_' + years_end + plot_style_type,\
                      dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Turnover
        plt.figure()
        turnover = backtest_output['turnover']
        plt.plot(turnover*100)
        plt.legend(returns.columns, frameon=False)
        plt.title(f"{self.settings['data_set']} - Turnover")
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_plot_path']+'\\' +self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] +'_turnover' + plot_style_type,\
              dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Allocations
        # plt.set_cmap('tab20c')
        allocations = backtest_output['allocations']
        for i in enumerate(allocations.keys()):
            tmp_allocations = allocations[i[1]].fillna(0)
            plt.figure()
            if self.settings['long_only_portfolio_weights']:
                plt.stackplot(tmp_allocations.index, tmp_allocations.T, colors=plt.get_cmap('tab20c').colors[:20])
            else:
                plt.plot(tmp_allocations)
            # plt.legend(allocations[i[1]].columns, frameon=False)
            plt.title(f"{self.settings['data_set']} - Portfolio Allocation for {i[1]}")
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.gcf().savefig(self.settings['results_plot_path']+'\\' +self.settings['data_set_name'] + '_' + self.settings['backtest_combination'] +'_zallocations_' + i[1].replace('/','') + '_' + plot_style_type,\
                  dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
