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
from aggregated_functions import annual_measures, total_measures, turnover_measures, hierarchical_ridge, bayesian_lasso, bayesian_elastic_net, truncted_normal, ledoit_wolf, frahm_memmel, tou_zhou, fama_french
from sklearn.linear_model import Lasso
import math


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
        def last_day_of_month(date):
            if date.month == 12:
                return date.replace(day=31)
            return date.replace(month=date.month+1, day=1) - dt.timedelta(days=1)
        # read return data
        return_data = pd.read_excel(self.settings['data_path']+self.settings['data_set']+'.xlsx')
        return_data['Date'] = pd.to_datetime(return_data['Date'], format='%Y%m')
        return_data['Date'] = return_data['Date'].apply(last_day_of_month)
        return_data.set_index('Date', inplace=True)
        return_data.index.name = None
        # read factor data
        factor_data = pd.read_excel(self.settings['data_path']+'FFF35.xlsx','5FF')
        factor_data['Date'] = pd.to_datetime(factor_data['Date'], format='%Y%m')
        factor_data['Date'] = factor_data['Date'].apply(last_day_of_month)
        factor_data.set_index('Date', inplace=True)
        factor_data.index.name = None
        factor_data = factor_data.reindex(return_data.index).fillna(0)
        # clean data
        return_data[return_data<=-99.99] = 0 # cut NaNs
        return_data = return_data/100 # decimal returns
        factor_data[factor_data<=-99.99] = 0 # cut NaNs
        factor_data = factor_data/100 # decimal returns
        # save data
        self.data['returns'] = return_data
        self.data['factor_returns'] = factor_data
        self.data['index'] = pd.DataFrame(return_data.mean(axis=1),columns=['Index'])
        self.data['dates'] = return_data.index
        return
    def calc_allocation(self, returns, opt_method, factors):
        ''' calc_allocation: Specify minimization algorithm and calculate weights
        - initial: initial weight guess
        '''
        # Check if data is complete (only use columns where there is actual data)
        returns = returns.loc[:, (returns.sum(axis=0) !=0)].copy()
        number_simulations = self.settings['number_simulations']

        if opt_method == 0: # 1/N portfolio

            strategy_name = '1/N'
            final_weights = np.ones((returns.shape[1], 1))/returns.shape[1]

        if opt_method == 1: # GMVP (no constraints)

            strategy_name = 'GMVP'
            y = np.array(returns.iloc[:,0])
            x = np.array(-returns.iloc[:,1:].subtract(y, axis=0))
            x = np.insert(x,0,1,axis=1)
            weights = np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))
            final_weights = np.append(1-weights.sum(),weights[1:]).round(8)


        if opt_method == 2:  # GMVP with Ridge tau=T/N

            strategy_name = 'Ridge'
            T, N = returns.shape
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)
            tau = T/N
            weights_tilde = np.linalg.pinv(np.eye(N+1)/tau+X.T.dot(X)).dot(X.T.dot(y))
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        if opt_method == 3:  # Empirical Bayes

            strategy_name = 'Empirical Bayes'
            T, N = returns.shape
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)
            tau = T/N
            weights_tilde = (tau/tau+1) * np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 4: # Hierarchical Ridge

            strategy_name = 'Hierarchical Ridge'
            N = returns.shape[1]
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)

            beta, sigma, tau = hierarchical_ridge(y, X, 1000)
            weights_tilde = beta.mean(axis=0)
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 5: # Bayesian Lasso

            strategy_name = 'Bayesian Lasso'
            N = returns.shape[1]
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)

            beta, sigma, invtau2, lambda_out = bayesian_lasso(y, X, number_simulations)
            weights_tilde = beta.mean(axis=0)
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 6: # Bayesian Elastic Net

            strategy_name = 'Bayesian Elastic Net'
            N = returns.shape[1]
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)

            beta, sigma, invtau2, lambda1_out, lambda2_out = bayesian_elastic_net(y, X, number_simulations)
            weights_tilde = beta.mean(axis=0)
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 7: # Truncted Normal

            strategy_name = 'Truncted Normal'
            N = returns.shape[1]
            w_0 = np.ones([N,1]) / N
            y = np.matmul(np.array(returns), w_0)
            X = np.repeat(y,N,axis=1) - np.array(returns)
            X = np.insert(X,0,1,axis=1)

            beta, sigma = truncted_normal(y, X, number_simulations, -1/N, 1)
            weights_tilde = beta.mean(axis=0)
            final_weights = weights_tilde[1:] + (1/N) * (1 - weights_tilde[1:].sum())

        elif opt_method == 8: # Ledoit & Wolf

            strategy_name = 'LW'
            weights, Sigma = ledoit_wolf(returns, 1)
            final_weights = weights

        elif opt_method == 9: # Frahm & Memmel

            strategy_name = 'FM'
            weights = frahm_memmel(returns, 2)
            final_weights = weights

        elif opt_method == 10: # Tou & Zhou

            strategy_name = 'TZ'
            weights = tou_zhou(returns, 10)
            final_weights = weights

        elif opt_method == 11: # Fama French

            strategy_name = 'FF'
            weights = fama_french(returns, factors)
            final_weights = weights

        elif opt_method == 12: # 1/vol portfolio

            strategy_name = '1/vol'
            final_weights = 1/np.std(returns)
            final_weights = final_weights / final_weights.sum()

        elif opt_method == 13: # GMVP (no short sales)

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

        elif opt_method == 14: # equal risk contribution portfolio

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
        allocation = pd.DataFrame(final_weights)
        allocation.index = returns.columns
        allocation = allocation.groupby(by=allocation.index).sum()
        allocation = allocation.fillna(0)
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
        all_dates = pd.DataFrame(self.data['returns'].loc[start:end].index).rename(columns={0: 'date'})

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
        all_dates.drop(all_dates[all_dates['dayofweek'] >= 5].index , inplace=True) # Drop weekends
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
        returns_all_ni = returns_all.copy().reset_index()
        for i in range(0, len(dates_reb), 1):
            # First get the timing right
            tmp_date = str(dates_reb[i]).split()[0]
            end_window = returns_all_ni[returns_all_ni['index'] == tmp_date].index.tolist()
            end_window = end_window[0] - 1
            start_window = end_window - window
            try:
                tmp_returns = returns_all.iloc[start_window:end_window, :]
                tmp_factor_returns = factor_returns.iloc[start_window:end_window, :]
            except (RuntimeError, ValueError, TypeError, NameError, IndexError):
                tmp_returns = returns_all.iloc[:end_window, :]
                tmp_factor_returns = factor_returns.iloc[:end_window, :]
            allocation_dates.append(tmp_date)
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
        all_alloc = allocations['weights'].unstack().replace(np.nan, 0)
        all_alloc.index = pd.to_datetime(all_alloc.index)
        all_alloc = all_alloc.reindex(ret.loc[self.settings['start_date']\
                                              :self.settings['end_date']].index).ffill()
        for i in range(1, len(dates)):
            cum_ret = (1+ret.loc[dates[i-1]:dates[i], all_alloc.columns]).cumprod()
            cum_ret = cum_ret.iloc[1:-1, :]
            all_alloc.loc[cum_ret.index, :] = all_alloc.loc[cum_ret.index, :]*cum_ret
        # cumulative performance after last rebalancing to end of the sample
        cum_ret = (1+ret.loc[dates[-1]:, all_alloc.columns]).cumprod()
        cum_ret = cum_ret.iloc[1:-1, :]
        all_alloc.loc[cum_ret.index, :] = all_alloc.loc[cum_ret.index, :]*cum_ret
        all_alloc = all_alloc.div(all_alloc.sum(axis=1),axis=0)
        return all_alloc, dates
    def backtest(self):
        ''' backtest: Returns the historical performance of the allocation
        '''
        # Data
        self.get_data()
        ret = self.data['returns']
        index = self.data['index']
        window = self.settings['window']
        length_year = self.settings['length_year']
        risk_aversion = self.settings['risk_aversion']
        N = ret.shape[1]

        # Allocation
        allocations = dict()
        returns = dict()
        performances = dict()
        annuals = dict()
        totals = pd.DataFrame()
        performances['returns'] = pd.DataFrame()
        performances['WD'] = pd.DataFrame()

        for i in enumerate(self.settings['opt_method']):

            allocations[i[1]], dates = self.complete_allocation(i[1])
            strategy_name = allocations[i[1]].columns.name
            returns[i[1]] = ret[list(allocations[i[1]].columns)].loc[self.settings['start_date']:self.settings['end_date'], :]
            weight_change = abs(allocations[i[1]]-allocations[i[1]].shift()).sum(axis=1)
            # substract costs
            for j in enumerate(dates):
                costs = self.settings['costs'] * weight_change[dates[j[0]]]
                returns[i[1]].loc[dates[j[0]], :] = returns[i[1]].loc[dates[j[0]], :] - costs
            performances['returns'].loc[:,strategy_name] = (allocations[i[1]].shift() * returns[i[1]]).sum(axis=1)
            performances['WD'].loc[:,strategy_name]  = abs(allocations[i[1]]-1/N).sum(axis=1)
            annuals[strategy_name] = annual_measures(performances['returns'].loc[:,strategy_name])
            print('Allocation for ' + allocations[i[1]].columns.name + ' is complete.')

        performances['AR'] = performances['returns'].rolling(window=window, min_periods=1).mean() * 100 * length_year
        performances['SD'] = performances['returns'].rolling(window=window, min_periods=1).std() * 100 * np.sqrt(length_year)
        performances['SR'] = performances['AR'].div(performances['SD'])
        performances['CE'] = performances['AR'] - (risk_aversion/2) * performances['SD']
        performances['RL'] = performances['SD'].mul(performances['AR'].loc[:,'1/N'].div(performances['SD'].loc[:,'1/N']),axis=0) - performances['AR']
        performances['DD'] = (1+performances['returns']).cumprod(axis=0)/(1+performances['returns']).cumprod(axis=0).cummax(axis=0)-1
        totals = total_measures(performances['returns'], length_year, risk_aversion)

        allocation = allocations[0].copy()
        all_rets = ret[list(allocation.columns)].loc\
            [self.settings['start_date']:self.settings['end_date'], :]
        weight_change = abs(allocation-allocation.shift()).sum(axis=1)
        # substract costs
        for i in enumerate(dates):
            costs = self.settings['costs']*weight_change[dates[i[0]]]
            all_rets.loc[dates[i[0]], :] = all_rets.loc[dates[i[0]], :] - costs
        # performance
        performance = (allocation.shift() * all_rets).sum(axis=1)
        performance = pd.concat([index, performance], axis=1).dropna()
        performance.columns = ['Index', 'Portfolio']
        try:
            performance = performance.iloc[np.argmax(np.where(performance.loc\
                                                              [:, 'Portfolio'] == 0))+1:, :]
        except (RuntimeError, ValueError, TypeError, NameError, IndexError):
            pass
        # turnover
        turnover, annual_turnover = turnover_measures(allocation, dates)
        # annual perfromances
        annuals_portfolio = annual_measures(performance.loc[:, 'Portfolio'])
        annuals_index = annual_measures(performance.loc[:, 'Index'])
        annuals_te = annual_measures(performance.loc[:, 'Portfolio'] - performance.loc[:, 'Index'])
        # put in structure to be saved
        backtest_output = dict()
        backtest_output['all_performances'] = performances
        backtest_output['annuals'] = annuals
        backtest_output['totals'] = totals
        backtest_output['performances'] = performances['returns']
        backtest_output['allocation'] = allocation
        backtest_output['annual_turnover'] = annual_turnover
        backtest_output['performance'] = performance
        backtest_output['daily_turnover'] = turnover
        backtest_output['annual_turnover'] = annual_turnover
        backtest_output['dates'] = dates
        backtest_output['annuals_portfolio'] = annuals_portfolio
        backtest_output['annuals_index'] = annuals_index
        backtest_output['annuals_te'] = annuals_te
        # plot
        if (self.settings['plot']) and (self.settings['start_date'] != self.settings['end_date']):
            self.plot_backtest(backtest_output)
        else:
            pass
        return backtest_output
    def plot_backtest(self, backtest_output, *years):
        ''' plot_backtest: Plot backtest figures
        '''
        plot_style_type = self.settings['plot_style_type']
        if len(years) == 0:
            years = str(dt.datetime.strptime(self.settings['start_date'],'%Y%m%d').year)
        else:
            years = years[0]
        try:
            years_end = dt.datetime.strftime(self.settings['end_date'],'%Y')
        except:
            years_end = str(dt.datetime.strptime(self.settings['end_date'],'%Y%m%d').year)
        # redates = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in backtest_output['dates']]
        # Data
        portfolio_returns = backtest_output['performance']
        portfolio_returns_all = backtest_output['performances']
        # Equity Lines
        fig = plt.figure()
        plt.plot(((1+portfolio_returns_all).cumprod(axis=0)*100))
        plt.legend(portfolio_returns_all.columns, frameon=False)
        plt.title('Performance')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\CumulativePerformance_' + years + '_' + years_end + plot_style_type,\
             dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Equity Lines for each year
        if self.settings['plot_perfromance_years']:
            for i in enumerate(portfolio_returns_all.index.year.unique()):
                fig = plt.figure()
                try:
                    first_obs = pd.DataFrame([0,0], index=portfolio_returns_all.columns).T#, columns=portfolio_returns[str(i[1])].index[0]-dt.timedelta(days=1))#pd.DataFrame(columns=portfolio_returns.columns) pd.DataFrame(0) -dt.timedelta(days=30)
                    tmp_portfolio_returns = pd.concat([first_obs, portfolio_returns_all[str(i[1])]],axis=0)
                    tmp_portfolio_returns.index[0] = pd.Timestamp(str(i[1])+'-01-01')
                except:
                    tmp_portfolio_returns = portfolio_returns_all[str(i[1])]
                    pass
                plt.plot(((1+tmp_portfolio_returns).cumprod()*100))
                plt.legend(portfolio_returns_all.columns, frameon=False)
                plt.title('Performance in '+str(i[1]))
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.gcf().savefig(self.settings['results_path']+'\\CumulativePerformance_' + str(i[1]) + plot_style_type,\
                     dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Hockey Stick
        fig = plt.figure()
        plt.scatter(x=portfolio_returns['Portfolio'], y=portfolio_returns['Index'])
        plt.title('Scatter Plot Returns - Portfolio vs. index')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel('Portfolio')
        plt.ylabel('Index')
        plt.gcf().savefig(self.settings['results_path']+'\\HockeyStick_' + years + '_' + years_end + plot_style_type,\
             dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Alpha
        fig = plt.figure()
        plt.plot(((1+portfolio_returns['Portfolio']-portfolio_returns['Index']).cumprod()*100))
        plt.title('Portfolio Alpha Performance vs. Index')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\AlphaPerformance_' + years + '_' + years_end + plot_style_type,\
             dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Daily alpha
        fig = plt.figure()
        plt.plot(1+portfolio_returns['Portfolio']-portfolio_returns['Index'])
        plt.title('Daily Difference between Portfolio and Index')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\AlphaDaily_' + years + '_' + years_end + plot_style_type,\
                     dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Rolling Annualized Volatility
        fig = plt.figure()
        window = self.settings['window']
        plt.plot(portfolio_returns_all.rolling(window=window,min_periods=1).std()*100*np.sqrt(self.settings['length_year']))
        plt.legend(portfolio_returns_all.columns, frameon=False)
        plt.title('Annualized Volatility')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\AnnualizedVola_' + years + '_' + years_end + plot_style_type,\
              dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Rolling Sharpe Raio
        fig = plt.figure()
        window = self.settings['window']
        plt.plot((portfolio_returns_all.rolling(window=window,min_periods=1).mean()/portfolio_returns_all.rolling(window=window,min_periods=1).std())*np.sqrt(self.settings['length_year']))
        plt.legend(portfolio_returns_all.columns, frameon=False)
        plt.title('Annualized Sharpe Ratio')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\AnnualizedSharpe_' + years + '_' + years_end + plot_style_type,\
              dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Turnover
        turnover_portfolio = backtest_output['daily_turnover']['turnover_value']
        fig = plt.figure()
        (turnover_portfolio*100).plot()
        plt.title('Portfolio Turnover')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\Turnover_' + years + '_' + years_end + plot_style_type,\
              dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Cum Turnover
        fig = plt.figure()
        ((1+turnover_portfolio).cumprod()*100).plot()
        plt.title('Cumulative Portfolio Turnover (Hedging)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\CumulativeTurnover_' + years + '_' + years_end + plot_style_type,\
              dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Max Drawdown
        fig = plt.figure()
        drawdown_portfolio = (1+portfolio_returns).cumprod(axis=0)/(1+portfolio_returns).cumprod(axis=0).cummax(axis=0)-1
        plt.plot(drawdown_portfolio*100)
        plt.title('Drawdown')
        plt.legend(['Index', 'Portfolio'], frameon=False)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\Drawdown_' + years + '_' + years_end + plot_style_type,\
              dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Allocation over time
        fig = plt.figure()
        plt.plot(backtest_output['allocation'])
        plt.title('Allocations over time')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend(backtest_output['allocation'].columns, frameon=False)#, bbox_to_anchor=(1, 0.5))
        plt.gcf().savefig(self.settings['results_path']+'\\Allocations_' + years + '_' + years_end + plot_style_type,\
                     dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
        # Correlation over time
        fig = plt.figure()
        plt.plot(portfolio_returns.rolling(window=window, min_periods=1).corr().iloc[:,1].unstack().iloc[:,0])
        plt.title('Correlation over time')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gcf().savefig(self.settings['results_path']+'\\Correlation_' + years + '_' + years_end + plot_style_type,\
                     dpi=200, bbox_inches='tight', pad_inches=0, transparent=False)
