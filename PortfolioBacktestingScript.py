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
import pandas_datareader

from backtest_functions import PortfolioBacktest
from aggregated_functions import *
warnings.simplefilter(action='ignore')
TIC = time.time()

# In[Plot Settings]:

try:
    plt.style.use('plots_colors.mplstyle')
except Exception:
    plt.style.use('ggplot')

# plt.style.use('tableau-colorblind10')
plt.rcParams['figure.figsize'] = 15, 4
plt.rcParams['figure.max_open_warning'] = 100

# from cycler import cycler
# line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
#                  cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
# marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
#                  cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
#                  cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
# plt.rc("axes", prop_cycle=line_cycler)

# In[Initialize Modules]:


def getParentDir(path, level=1):
    ''' getParentDir: Get parents path '''
    return os.path.normpath(os.path.join(path, *([".."] * level)))


PATH = getParentDir(__file__, 1)
DATA_PATH = os.path.join(PATH, 'data')
RESULTS_PATH = os.path.join(PATH, 'results')
PB = PortfolioBacktest()
BACKTEST = {}

try:  # Create Results Path
    os.makedirs(RESULTS_PATH)
except FileExistsError:
    pass

# In[Settings]:

# pandas_datareader.famafrench.get_available_datasets()

all_datasets = {
                '5_Industry_Portfolios': '5_FF_Ind',
                '30_Industry_Portfolios': '30_FF_Ind',
                '49_Industry_Portfolios': '49_FF_Ind',
                # '6_Portfolios_2x3': '6_FF_PF',
                # # # '25_Portfolios_5x5': '25_FF_PF',
                '100_Portfolios_10x10': '100_FF_PF',
                # '25_LTOTMKUS': '25_DS_US',
                # # '50_LTOTMKUS': '50_DS_US',
                # # '100_LTOTMKUS': '100_DS_US',
                # # '250_LTOTMKUS': '250_DS_US',
                # # '500_LTOTMKUS': '500_DS_US',
                '25_LNYSEALL': '25_DS_NYSE',
                # # '50_LNYSEALL': '50_DS_NYSE',
                '100_LNYSEALL': '100_DS_NYSE',
                '250_LNYSEALL':'250_DS_NYSE',
                '500_LNYSEALL': '500_DS_NYSE',
                }

dataset_names_dict = {
    'FF': 'Fama French',
    'DS': 'Refinitiv',
    'Ind': 'Industry',
    'PF': 'Size to B/M',
    'US': 'Total US Market',
    'NYSE': 'NYSE Exchange'
    }

all_models = [
    '1/N',
    # '1/vol',
    'GMVP',
    'EmpBayes',
    'Ridge',
    # # 'HierRidge',
    # # 'BayLasso',
    'Lasso',
    # # # 'BayElasticNet',
    'ElasticNet',
    'Truncted Normal',
    'LW',
    'FF',
    'FM',
    'TZ'
    ]


for data_i in all_datasets:

    print('')
    print(f'Backtest for {data_i}')
    print('')

    PB.settings['data_set'] = data_i
    PB.settings['data_set_name'] = all_datasets[data_i]
    PB.settings['path'] = PATH
    PB.settings['data_path'] = DATA_PATH

    # settings for optimization
    PB.settings['opt_method'] = all_models
    PB.settings['lower'] = 0
    PB.settings['upper'] = 1
    PB.settings['risk_aversion'] = 1

    # backtest settings
    PB.settings['start_date'] = '19900101'
    PB.settings['end_date'] = '20181231'
    PB.settings['long_only_portfolio_weights'] = False
    PB.settings['rebalancing_period'] = 'months'
    PB.settings['rebalancing_frequency'] = 1
    PB.settings['costs'] = 0.005
    PB.settings['min_weight_change'] = 0.0
    PB.settings['window'] = 60
    PB.settings['forward_window'] = 60
    PB.settings['correlation_threshold'] = 0.95
    PB.settings['length_year'] = 12
    PB.settings['round_decimals'] = 4
    PB.settings['plot'] = True
    PB.settings['plot_performance_years'] = False
    PB.settings['plot_style_type'] = '.svg'
    PB.settings['number_simulations'] = 10
    PB.settings['update_data'] = False
    PB.settings['p_values_bootstrapped'] = False
    PB.settings['plot_rolling'] = False
    PB.settings['normalized_returns'] = False

    # create results folder for each backtest combination
    pf_weights = 'ls'
    if PB.settings['long_only_portfolio_weights']:
        pf_weights = 'lo'

    backtest_combination = (f"{pf_weights}"
                            + f"_{PB.settings['window']}"
                            + f"_{PB.settings['start_date'][:4]}"
                            + f"_{PB.settings['end_date'][:4]}"
                            )
    PB.settings['backtest_combination'] = backtest_combination
    PB.settings['results_path'] = os.path.join(RESULTS_PATH,
                                               backtest_combination)
    PB.settings['results_plot_path'] = os.path.join(
        RESULTS_PATH, backtest_combination, 'plots')
    PB.settings['results_tex_path'] = os.path.join(
        RESULTS_PATH, backtest_combination, 'tex')
    PB.settings['results_data_path'] = os.path.join(
        RESULTS_PATH, backtest_combination, 'data')

    try:  # Create Results Path
        os.makedirs(PB.settings['results_path'])
    except FileExistsError:
        pass

    try:  # Create Results Path
        os.makedirs(PB.settings['results_plot_path'])
    except FileExistsError:
        pass

    try:  # Create Results Path
        os.makedirs(PB.settings['results_tex_path'])
    except FileExistsError:
        pass

    try:  # Create Results Path
        os.makedirs(PB.settings['results_data_path'])
    except FileExistsError:
        pass

    # In[Backtest]:
    BACKTEST[data_i] = PB.backtest()

# In[Tables]

all_tables = {}
all_p_values = {}
all_measures = ['MeanReturn', 'Volatility', 'MDD', 'Sharpe', 'Return Loss',
                'Certainty Equivalent', 'MAD']

# Latex
decimal_format_dict = {'MeanReturn': '.1f',
                       'Volatility': '.1f',
                       'MDD': '.1f',
                       'Sharpe': '.2f',
                       'Return Loss': '.1f',
                       'Certainty Equivalent': '.2f',
                       'MAD': '.2f'
                       }

measure_dict = {'MeanReturn': "portfolio return",
                'Volatility': "portfolio volatility",
                'MDD': "maximum drawdown",
                'Sharpe': "Sharpe ratio",
                'Return Loss': "return loss",
                'Certainty Equivalent': "certainty equivalent",
                'MAD': "mean absolute weight deviations"
                }
measure_dict

all_models = BACKTEST[next(iter(all_datasets.keys()))]['totals'].columns

for measure_i in all_measures:

    tmp_measure = pd.DataFrame(index=all_models)
    tmp_p_values = pd.DataFrame(index=all_models)

    for data_j in list(all_datasets.keys()):

        tmp_measure[all_datasets[data_j]] = BACKTEST[data_j]['totals'].T[measure_i]
        tmp_p_values[all_datasets[data_j]] = BACKTEST[data_j]['p_values'].T[measure_i]

    # Create multicolum index for latex output
    tuples = [(dataset_names_dict[col.split('_')[1]],
                dataset_names_dict[col.split('_')[2]],
                col.split('_')[0]) for col in tmp_measure.columns]
    multiindex = pd.MultiIndex.from_tuples(tuples, names=['\\textbf{Vendor}', '\\textbf{Dataset}', '\\textbf{Asset Universe}'])

    tmp_measure.columns = multiindex
    tmp_p_values.columns = multiindex

    all_tables[measure_i] = tmp_measure
    all_p_values[measure_i] = tmp_p_values

    print('')
    # print(measure_i)
    print('')

    if PB.settings['long_only_portfolio_weights']:
        long_indictor = 'long-only'
    else:
        long_indictor = 'long-short'

    # Latex
    decimal_format = decimal_format_dict[measure_i]

    # Generate Latex
    if measure_i in ['Volatility', 'MDD',  'MAD', 'Return Loss']:
        highlight_type = 'min'
    else:
        highlight_type = 'max'

    # Create caption for table
    data_dates = [dt.datetime.strftime(i, "%B %Y") for i in
                  list(BACKTEST[data_j]['performance']['returns'].index[[0, -1]])]

    caption_text = (f'Out-of-sample mean results {"(in percent) "*(measure_i != "Sharpe")}for the '
        + f'{measure_dict[measure_i]} relative to the 1/N portfolio using various '
        + f'data sets with h = {PB.settings["window"]} months estimation window '
        + f'size for a {long_indictor} portfolio with rebalancing after {PB.settings["rebalancing_frequency"]} '
        + f'period{"s"*(PB.settings["rebalancing_frequency"]>1)}. The evaluation '
        + f'sample is from {data_dates[0]} to {data_dates[1]}.\n\\vspace{{0.0em}}'
        )
    # Create note text for table
    note_text = (f'\\vspace{{0.6em}}\\\\\n{{\\footnotesize \\textit{{Note:}} '
        + f'The table reports out-of-sample mean results (in percent) for the '
        + f'{measure_dict[measure_i]} for various data sets in a rolling window ' 
        + f'one-step ahead {long_indictor} portfolio optimization. For all datasets, portfolio '
        + f'returns are net of transaction costs {int(PB.settings["costs"]*100*100)} '
        + f'basis points per trade. The bold number in each column indices the '
        + f'{"smallest"*(highlight_type == "min")+"largest"*(highlight_type == "max")} '
        + f'value in each column. For the Refinitive data, we follow the methodology of '
        + f'\\citet{{denard2022}} in section 5.2 on page 7, and choose the assets with the '
        + f'highest market capitalization while excluding assets with pairwise correlations '
        + f'higher than {PB.settings["correlation_threshold"]}. '
        )
    # if measure_i == 'Certainty Equivalent':
    #    note_text += f'We assume a risk aversion of $\gamma={PB.settings["risk_aversion"]}$ to calculate the certainty equivalents. '
    #    note_text += f'To test the difference in certainty equivalents, we use the methodology described by \\citet{{demiguel2009}} on page 1929. '
    #    note_text += f'One/two/three asterisks denote rejection of the null hypothesis of a smaller or equal certainty equivalent than 1/N at the ten/five/one percent test level.'
       
    # if measure_i == 'Volatility':
    #    note_text += f'To test the difference in standard deviations, we use a bootstrap approach similar to the methodology described by \\citet{{ledoit2008}}. '
    #    note_text += f'One/two/three asterisks denote rejection of the null hypothesis of a smaller or equal standard deviation than 1/N at the ten/five/one percent test level.'
       
    # if measure_i == 'Sharpe':
    #    note_text += f'To test the difference in Sharpe ratios, we use a bootstrap approach similar to the methodology described by \\citet{{ledoit2011}}. '
    #    note_text += f'One/two/three asterisks denote rejection of the null hypothesis of a smaller or equal Sharpe ratio than 1/N at the ten/five/one percent test level.'
       
    
    note_text += '}}'
    note_text += '\n\\end{table}'
    
    column_format = 'l'+''.join(['c']*all_tables[measure_i].shape[1])


    df_highlighted = (highlight_max_with_pvals(
        all_tables[measure_i], all_p_values[measure_i],
        highlight_type=highlight_type, decimal_format=decimal_format, bold=True)
        .to_latex(caption=caption_text,
                  label=f"tab:{backtest_combination}_{measure_i.replace(' ', '')}",
                  escape=False,
                  index=True,
                  index_names=True,
                  sparsify=True,
                  multirow=True,
                  multicolumn=True,
                  multicolumn_format='c',
                  position='p',
                  bold_rows=True,
                  column_format=column_format
                  )
        )

    # adjust table for fromating


    # \cmidrule(r{1em}l){3-6}\cmidrule(l{1em}){7-11}
    df_highlighted_list = df_highlighted.splitlines()
    df_highlighted_list.insert(1, '\\fontsize{10}{18}\\selectfont{')

    for i in [1, 3]:

        tmp_index = df_highlighted_list.index('\\toprule')+i
        tmp_string = df_highlighted_list[tmp_index]

        multicol_len = [str(item.split('multicolumn{')[1][0])
                        if '\\multicolumn' in item else '1'
                        for item in tmp_string.split('&')[1:]
                        ]

        # multicol_len = [i+1 for i in range(len(tmp_string))
        #                 if tmp_string.endswith('multicolumn{', 0, i + 1)]
        # multicol_len = [tmp_string[i] for i in multicol_len]
        cmid_string = ''
        for j in enumerate(multicol_len):
            if j[0] == 0:
                tmp_start_index = 2
            cmid_string += '\cmidrule(r{0.1em}l){'+str(tmp_start_index)+'-'+str(tmp_start_index+int(j[1])-1)+'}'
            tmp_start_index = tmp_start_index + int(j[1])

        df_highlighted_list.insert(tmp_index+1, cmid_string)

    df_highlighted = '\n'.join(df_highlighted_list)
    df_highlighted = df_highlighted.replace('\end{table}', note_text)

    # Wrap the LaTeX output for table to span the entire textwidth
    df_highlighted = df_highlighted.replace('\\begin{tabular}{'+column_format+'}', '\\begin{tabularx}{\\textwidth}{'+column_format.replace('l','X')+'}')
    df_highlighted = df_highlighted.replace('\\end{tabular}', '\\end{tabularx}')

    # Replace with real citet
    df_highlighted = df_highlighted.replace('\\textbf{LW}', '\\textbf{\\citet{{ledoit2003}}}')
    df_highlighted = df_highlighted.replace('\\textbf{FM}', '\\textbf{\\citet{{frahm2010b}}}')
    df_highlighted = df_highlighted.replace('\\textbf{TZ}', '\\textbf{\\citet{{tu2011}}}')
    df_highlighted = df_highlighted.replace('\\textbf{FF}', '\\textbf{\\citet{{fama2015}}}')

    tex_file_name = f"{backtest_combination}_{measure_i.replace(' ', '')}.tex"
    with open(os.path.join(PB.settings['results_tex_path'],
                           tex_file_name), "w") as f:
        f.write(df_highlighted)

    # print(f'LaTeX table saved to {tex_file_name}')
    # print('')
    # print(df_highlighted)

print(all_tables['Sharpe'].round(2))
print(all_tables['Volatility'].round(2))

# In[Create html file for all plots]:
    
def generate_html(images_folder, file_name, datasets, scenario_name='None'):
    """Generate html grid for plots."""
    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(images_folder)
                    if f.endswith(('.svg', '.jpeg', '.png', '.gif'))]
    
    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Grid and Table</title>
        <style>
            .image-container {
                display: flex;
                flex-wrap: wrap;
                max-width: 2000px; /* Limit maximum width */
                margin: 0 auto; /* Center the grid */
                padding: 0px;
            }
            .image-item {
                flex: 0 0 31%; /* Adjusted width for spacing bw. images */
                box-sizing: border-box;
                padding: 0px;
                min-width: 300px; /* Set a minimum width for the image */
                margin: 1%; /* Add some margin for spacing bw. images */
            }
            .image-item img {
                width: 100%;
                display: block;
            }
        </style>
    </head>
    <body>"""
    html_content += '<h1>' + scenario_name + '</h1>'
    html_content += '<div class="image-container">'
    
    # html_content = html_content.replace('{', '{{').replace('}', '}}')
    
    # Iterate through images and create HTML img elements
    for dataset_i in datasets:
        
        image_files = [f for f in os.listdir(images_folder)
                        if f.endswith(('.svg', '.jpeg', '.png', '.gif'))
                        and f.startswith((datasets[dataset_i]))]
        html_content += "</div>"
        html_content += f"<hr>{dataset_i}<hr>"
        html_content += '<div class="image-container">'
        
    
        for image_file in image_files:    
            img_path = os.path.join(images_folder, image_file)
            # img_title = os.path.splitext(image_file)[0]  # file name as title
            html_content += (
                f"<div class='image-item'><img src='{img_path}'></div>")
            html_content += """
            """
    
    html_content += """
        </div>
    </body>
    </html>"""
    
    # Write HTML content to a file
    with open(file_name, "w") as html_file:
        html_file.write(html_content)

target_file = os.path.join(PB.settings['results_path'], 'results_overview.html')
generate_html(PB.settings['results_plot_path'], target_file, all_datasets, scenario_name=f"Portfolio Backtest Results for {backtest_combination} Combination")

# In[End of Script]:
print('\n The code execution finished in %s seconds.' % round(time.time() - TIC,1))
