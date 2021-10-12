# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:42:42 2021

@author: Frey
"""
# In[Portfolio Allocation Backtest]:

# In[Load Libararies]:

import datetime as dt
import warnings
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import itertools as it

from backtest_functions import PortfolioBacktest
warnings.simplefilter(action='ignore')
TIC = time.time()

# In[Plot Settings]:

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 15, 4
plt.rcParams['figure.max_open_warning'] = 100

# In[Initialize Modules]:

def getParentDir(path, level=1):
    ''' getParentDir: Get parents path '''
    return os.path.normpath(os.path.join(path, *([".."] * level)))

PATH = getParentDir(__file__,1)
DATA_PATH = PATH + '\\data\\'
RESULTS_PATH = PATH + '\\results\\'
PB = PortfolioBacktest()

# In[Settings]:

PB.settings['data_set'] = '5_Industry'
PB.settings['path'] = PATH
PB.settings['data_path'] = DATA_PATH
PB.settings['results_path'] = RESULTS_PATH

# settings for optimization
PB.settings['opt_method'] = [0, 1, 2, 3]#, 4, 5, 6]#, 7, 8, 9, 10, 11]
PB.settings['lower'] = 0.0
PB.settings['upper'] = 0.1
PB.settings['risk_aversion'] = 1

# backtest settings
PB.settings['start_date'] = '20110101'
PB.settings['end_date'] = '20211231'
PB.settings['rebalancing_period'] = 'months'
PB.settings['rebalancing_frequency'] = 1
PB.settings['costs'] = 0.005
PB.settings['min_weight_change'] = 0.0
PB.settings['window'] = 60
PB.settings['length_year'] = 12
PB.settings['plot'] = False
PB.settings['plot_perfromance_years'] = False
PB.settings['plot_style_type'] = '.png'
PB.settings['number_simulations'] = 100
PB.settings['update_data'] = False

# In[Backtest]:
BACKTEST = PB.backtest()

# In[End of Script]:
print('\n The code execution finished in %s seconds.' % round(time.time() - TIC,1))
