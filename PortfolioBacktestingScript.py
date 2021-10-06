# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:42:42 2021

@author: Frey
"""
# In[Portfolio Allocation Backtest]: 
5
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
PB.settings['plot_style_type'] = '.png'


# settings for optimization
PB.settings['opt_method'] = 12
PB.settings['lower'] = 0.0
PB.settings['upper'] = 0.1

# backtest settings
PB.settings['start_date'] = '20000101'
PB.settings['end_date'] = '20211231'
PB.settings['rebalancing_period'] = 'months'
PB.settings['rebalancing_frequency'] = 6
PB.settings['costs'] = 0.0005
PB.settings['min_weight_change'] = 0.0
PB.settings['window'] = 120
PB.settings['length_year'] = 250
PB.settings['plot'] = True
PB.settings['update_data'] = False

# In[Backtest]: 
BACKTEST = PB.backtest()