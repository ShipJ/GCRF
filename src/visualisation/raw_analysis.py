import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
plt.rcParams['figure.facecolor']='white'
from src.config import config


country = config.get_country()
cdr = config.get_master_cdr(country, 'bts').dropna()
cdr['Vol'] = cdr['Vol']/1000000.0
cdr['Log_vol'] = np.reciprocal(cdr['Vol']+1)

col_list = ["cool blue", "gunmetal"]
col_list_palette = sb.xkcd_palette(col_list)
sb.set_palette(col_list_palette)

# fig, axs = plt.subplots(figsize=(12, 9))
# sb.regplot('CellTowerID', 'Vol', data=cdr, fit_reg=False, ax=axs, marker='x')
# cdr['Sort_vol'] = sorted(cdr['Vol'])
# sb.regplot('CellTowerID', 'Sort_vol', data=cdr, fit_reg=False, ax=axs, marker='x')
# axs.set(xlabel='Cell Tower ID', ylabel='Call Volume per Cell Tower (millions)', title='Ivory Coast')
# plt.show()
# fig2, axs2 = plt.subplots(figsize=(12, 9))
# sb.regplot('CellTowerID', 'Log_vol', data=cdr, fit_reg=False, ax=axs2, marker='x')
# cdr['Sort_log_vol'] = sorted(cdr['Log_vol'])
# sb.regplot('CellTowerID', 'Sort_log_vol', data=cdr, fit_reg=False, ax=axs2, marker='x')
# axs2.set(xlabel='Cell Tower ID', ylabel='1/(Call Volume per Cell Tower (millions))', title='Ivory Coast')
# plt.show()
# fig3, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 9))
# sb.distplot(cdr['Vol'], ax=ax1, bins=12)
# sb.distplot(cdr['Log_vol'], ax=ax2, bins=12)
# ax1.set(xlabel='Call Volume (millions)', ylabel='Frequency', title='Ivory Coast')
# ax2.set(xlabel='1/(Call Volume (millions))', ylabel='Frequency', title='Ivory Coast')
# sb.plt.show()


fig, axs = plt.subplots(figsize=(12, 9))
dhs = config.get_master_dhs(country).dropna()
dhs['sortDeathRate'] = sorted(dhs['DeathRate'])
dhs['logdeathrate'] = np.log(dhs['DeathRate']+1)
dhs['sortlogdeathrate'] = sorted(dhs['logdeathrate'])

sb.regplot('Adm_4', 'DeathRate', data=dhs, fit_reg=False, ax=axs, marker='x')
sb.regplot('Adm_4', 'sortDeathRate', data=dhs, fit_reg=False, ax=axs, marker='x')
# sb.regplot('Adm_4', 'logdeathrate', data=dhs, fit_reg=False, ax=axs, marker='x')
# sb.regplot('Adm_4', 'sortlogdeathrate', data=dhs, fit_reg=False, ax=axs, marker='x')

plt.show()







