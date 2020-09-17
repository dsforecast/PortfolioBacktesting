def rebalancing_dates(dates, period, frequency, method, position, day):
    '''
    rebalancing_dates: calculte rebalancing dates for given vector of dates in time series
    Input:
    - dates: list of dates, e.g. RETURNS.index
    - period: string, e.g. days, weeks, months, years
    - frequency: scalar, gives observations frequency, e.g. every day, every second day etc.
    - method:  scalar = 1 for specifying which element in period, e.g. first, second,... or last etc.
                      = 2 for specifying which day of the week, Monday = 0, Friday = 4. 
    - position: scalar, which day to choose
    - day: scalar, Monday = 0, Friday = 4. 
    Output:
    - rebalancing_dates: vector of datetime objects included in dates input
    Requirments:
    import datetime as dt
    import pandas as pd
    '''

    all_dates = pd.DataFrame(dates).rename(columns={0: 'date'})
    
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
        all_dates = all_dates.groupby('period').filter(lambda x: x.shape[0] >= abs(position)+1)
        rebalancing_dates = all_dates.groupby('period').apply(lambda x: (x.iloc[position]))
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
