import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
plt.rcParams['figure.facecolor']='white'
from src.config import config


# country = config.get_country()
# cdr = config.get_master_cdr(country, 'bts').dropna()
# cdr['Vol'] = cdr['Vol']/1000000.0

# fig, axs = plt.subplots()
# sb.regplot('CellTowerID', 'Vol', data=cdr, fit_reg=False, ax=axs)
# cdr['Vol'] = sorted(cdr['Vol'])
# sb.regplot('CellTowerID', 'Vol', data=cdr, fit_reg=False, ax=axs)
# axs.set(xlabel='Cell Tower ID', ylabel='Call volume per Cell Tower (millions)')

# Four hls color palette (left)
# col_list = ["cool blue", "warm grey"]

#, "gunmetal", "dusky blue", "cool blue", "deep teal", ,

# sb.palplot(sb.xkcd_palette(col_list))
# col_list_palette = sb.xkcd_palette(col_list)
# sb.set_palette(col_list_palette)
#
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
#
# country = config.get_country()
# cdr = config.get_master_cdr(country, 'bts').dropna()
# cdr['Vol'] = cdr['Vol']/1000000.0
# sb.regplot(x=cdr['CellTowerID'], y=cdr['Vol'], ax=ax1, fit_reg=False)
# cdr['Vol'] = sorted(cdr['Vol'])
# sb.regplot(x=cdr['CellTowerID'], y=cdr['Vol'], ax=ax1, fit_reg=False)
# sb.distplot(cdr['Vol'], ax=ax3).set(xlim=(0, 6), ylim=(0, None))
#
#
# country = config.get_country()
# cdr = config.get_master_cdr(country, 'bts').dropna()
# cdr['Vol'] = cdr['Vol']/1000000.0
# sb.regplot(x=cdr['CellTowerID'], y=cdr['Vol'], ax=ax2, fit_reg=False)
# cdr['Vol'] = sorted(cdr['Vol'])
# sb.regplot(x=cdr['CellTowerID'], y=cdr['Vol'], ax=ax2, fit_reg=False)
# sb.distplot(cdr['Vol'], ax=ax4).set(xlim=(0, 20), ylim=(0, 0.3))
#
#
# ax1.set(xlabel='Call Volume (millions)', ylabel='Frequency', title='Ivory Coast')
# ax2.set(xlabel='Call Volume (millions)', ylabel='Frequency', title='Senegal')
# ax3.set(xlabel='Call Volume (millions)', ylabel='Frequency', title='Ivory Coast')
# ax4.set(xlabel='Call Volume (millions)', ylabel='Frequency', title='Senegal')
# sb.plt.show()


country = config.get_country()
dhs = config.get_master_dhs(country).dropna()
fig, (ax1, ax2) = plt.subplots(ncols=2)

sb.regplot(x=dhs['Adm_4'], y=dhs['BloodPosRate'], ax=ax1, fit_reg=False)
dhs['BloodPosRate'] = sorted(dhs['BloodPosRate'])
sb.regplot(x=dhs['Adm_4'], y=dhs['BloodPosRate'], ax=ax2, fit_reg=False)
sb.plt.show()











