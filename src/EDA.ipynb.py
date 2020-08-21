# %% markdown
# # real EDA
#   > pls use r environment
#   > do: conda activate r
#

#%%codecell
!conda env list

# %% markdown
# ## general import and setup
#
# The `R` interfaces can be disabled if not needed.


# %% codecell

import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt;
from matplotlib import dates as mdates;
import seaborn as sns;

sns.set_context("notebook", font_scale=1.5)
# R interfaces, can be disabled if not needed
# import rpy2;
# %load_ext rpy2.ipython
#
# import rpy2.robjects as ro;
#
# from rpy2.robjects import pandas2ri;
# from rpy2.robjects import default_converter
# from rpy2.robjects.conversion import localconverter;
#
# from rpy2.robjects.packages import importr;
#
# utils = importr("utils");
# grdevices = importr('grDevices');
# infenergy = importr('infenergy');
#
#
# def help_r( python_name ):
#     print( str( utils.help( python_name.replace('_','.') ) ) );

# %% markdown
# # main EDA
#
# remember ipykernel can do basic shell like `cd`, `ls`, etc
# as long as you append `!` before it.
#
# e.g.
#   - `!echo test`,
#   - `!echo {data} | grep -E '\d+:\d+:'`
#
# note that {data} is a python variable
#
# Also magic commands equilivant of them via appending `%`
#
# _note that this is fine if automagic is enabled but must add `%` if running block_
#
# e.g.
#  - %pwd
#  - pwd ../misc
#  - %R example(get.inf.meter.data)


# %% codecell

combined_data_with_nans = pd.read_csv('/Users/lautinyeung/Desktop/work/DS MSc/THESIS/code/data/preprocessed/combined/combined_data_with_nans.csv', index_col='OB_TIME')
combined_data_with_nans.index = pd.to_datetime(combined_data_with_nans.index).tz_convert(tz='GMT')

combined_data_with_nans.info()
combined_data_with_nans.head()

plt.figure(figsize=(10,8))
ax = sns.heatmap(
    combined_data_with_nans.dropna().corr(),
    cmap=plt.cm.coolwarm,
    vmax=1.0,
    linewidths=0.1,
    linecolor='white',
    square=True,
    annot=True,
    annot_kws={"size": 10}
)

# hided, too messy.  basically the heatmap but with kde of each feature + points projected into each 2d dimension
# pp = sns.pairplot(combined_data_with_nans.dropna(),
#                   height=1.8, aspect=1.2,
#                   plot_kws=dict(edgecolor="k", linewidth=0.5),
#                   diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots
#                   diag_kind="kde") # use "kde" for diagonal plots
#
#
# combined_data_with_nans['WIND_DIRECTION'].plot(kind='kde')

plt.show()


# %%markdown
# ## removed
# Parrallel coordinates is not common and not intuitive to interpret for majority of readers

# %%codecell
# from sklearn.preprocessing import StandardScaler
#
# ss = StandardScaler()
# combined_data_with_nans_scaled = ss.fit_transform(combined_data_with_nans)
# combined_data_with_nans_scaled = pd.DataFrame(combined_data_with_nans_scaled, columns=combined_data_with_nans.columns)
# combined_data_with_nans_scaled.index = combined_data_with_nans.index
# combined_data_with_nans_scaled.head()
#
#
# # %%markdown
#
# # %%codecell
#
# from pandas.plotting import parallel_coordinates
#
# fig = plt.figure(figsize=(20, 8))
# title = fig.suptitle("Parallel Coordinates", fontsize=18)
# fig.subplots_adjust(top=0.93, wspace=0)
#
#
# pc = parallel_coordinates(combined_data_with_nans_scaled.dropna().assign(dummy_class='dummy_class'),
#                           'dummy_class')





# %%markdown
# rip

# %%codecell
fig, axs = plt.subplots(3, 2, figsize=(15,15))


combined_data_with_nans['bin_kwh']['2014-4-30':'2015-4-30'].plot(ax=axs[0,0])
axs[0,0].set_title('1 year')

combined_data_with_nans['bin_kwh']['2014-4-30':'2014-11-20'].plot(ax=axs[0,1])
axs[0,1].set_title('6 months')

combined_data_with_nans['bin_kwh']['2014-4-30':'2014-7-29'].plot(ax=axs[1,0])
axs[1,0].set_title('3 months quarter')

combined_data_with_nans['bin_kwh']['2014-4-30':'2014-5-13'].plot(ax=axs[1,1])
axs[1,1].set_title('2 weeks')

combined_data_with_nans['bin_kwh']['2014-4-30':'2014-4-30'].plot(ax=axs[2,0])
axs[2,0].set_title('1 day')

fig.tight_layout()
plt.show()



# %%markdown
# ## Resampling daily and weekly
# bin_kwh are diffs, resample should go by sum:
# sum( diffs )
#
# = c<sub>n</sub> - c<sub>n-1</sub> + c<sub>n-1</sub> - c<sub>n-2</sub> .... c<sub>3</sub> - c<sub>2</sub> + c<sub>2</sub> - c<sub>1</sub>
#
# = c<sub>n</sub> - c<sub>1</sub>
#
# Weather feature are most useful to average.
# Taking last or max wont work.
#
# Note for feature type while averaging, default to mean except:
#
# WMO_HR_SUN_DUR: hours of sunlight duration within that hour.  USE SUM.
#
# [WIND_SPEED, WIND_DIRECTION]: self-explainatory.  USE SPECIAL MEAN ref. research_gate_avg_wind
#
# PRCP_AMT: self-explainatory. recorded hourly in mm.  USE SUM.
#
# OTHER:
# CLD_BASE_HT: height of clouds, maybe unreliably recorded.  Data seems rounded but unclear.
# Taking means seems to change the distribution a lot, doesnt quite match.  Logically this should be MEANed.
# %%codecell

combined_data_with_nans['AIR_TEMPERATURE'][combined_data_with_nans.isna()['AIR_TEMPERATURE']]



cols_to_sum  = np.array(['bin_kwh', 'WMO_HR_SUN_DUR', 'DRV_HR_SUN_DUR','PRCP_AMT'])
cols_winds   = np.array(['WIND_SPEED', 'WIND_DIRECTION'])
cols_to_mean = np.array([ col for col in combined_data_with_nans.columns if col not in np.append(cols_to_sum, cols_winds)])

len(   np.append(np.append(cols_to_sum, cols_winds), cols_to_mean)   )
len( combined_data_with_nans.columns )

part_mean = combined_data_with_nans[ cols_to_mean ].resample('D').mean();
part_mean.shape

part_sum = combined_data_with_nans[ cols_to_sum ].resample('D').sum()
part_sum.shape

#see ref. research_gate_avg_wind
def to_wind_avg( data ):
    sp = data['WIND_SPEED'];
    dir = data['WIND_DIRECTION'];

    u = -sp * np.sin( np.pi * dir / 180 )
    v = -sp * np.cos( np.pi * dir / 180 )

    mean_u = np.mean(u);
    mean_v = np.mean(v);

    flow = lambda deg: 180 if deg < 180 else -180;

    mean_sp  = ( mean_u**2 + mean_v**2 ) ** 0.5
    arctans_deg  = np.arctan2( mean_u, mean_v ) * 180 / np.pi
    mean_dir = arctans_deg + flow(arctans_deg);

    return pd.Series([mean_sp, mean_dir], index=['WIND_SPEED','WIND_DIRECTION'])

part_winds = combined_data_with_nans[ cols_winds ].groupby( pd.Grouper(freq='1d') ).apply( to_wind_avg )
part_winds.shape

combined_data_with_nans_daily = pd.concat([part_mean, part_sum, part_winds], axis=1)
combined_data_with_nans_daily = combined_data_with_nans_daily[ combined_data_with_nans.columns ] #unify column ordering
combined_data_with_nans_daily.shape

combined_data_with_nans_daily.isna()['bin_kwh'].value_counts()

combined_data_with_nans_daily.head()

# recording at first day started out at 18:00, drop this day for daily
combined_data_with_nans_daily = combined_data_with_nans_daily.iloc[1:]

combined_data_with_nans_daily.head()





part_mean = combined_data_with_nans[ cols_to_mean ].resample('W').mean();
part_mean.shape

part_sum = combined_data_with_nans[ cols_to_sum ].resample('W').sum()
part_sum.shape

part_winds = combined_data_with_nans[ cols_winds ].groupby( pd.Grouper(freq='1w') ).apply( to_wind_avg )
part_winds.shape

combined_data_with_nans_weekly = pd.concat([part_mean, part_sum, part_winds], axis=1)
combined_data_with_nans_weekly = combined_data_with_nans_weekly[ combined_data_with_nans.columns ] #unify column ordering
combined_data_with_nans_weekly.shape

combined_data_with_nans_weekly.isna()['bin_kwh'].value_counts()

combined_data_with_nans_weekly.head()

# recording at first day started out at 18:00, still significant at weekly, drop first week for weekly
combined_data_with_nans_weekly = combined_data_with_nans_weekly.iloc[1:]

combined_data_with_nans_weekly.head()

combined_data_with_nans.shape
combined_data_with_nans_daily.shape
combined_data_with_nans_weekly.shape

combined_data_with_nans_weekly.index.to_series().dt.dayofweek


sns.kdeplot(data=combined_data_with_nans['CLD_BASE_HT']['2015-1'], bw=10)
sns.kdeplot(data=combined_data_with_nans_daily['CLD_BASE_HT']['2015-1'], bw=6)
combined_data_with_nans['CLD_BASE_HT'].value_counts()

# %%markdown
# # Labels
# holiday:
#  - hourly
#  - daily
#  - weekly
#
# weekday / weekend:
#  - hourly
#  - daily
#
# no wind:
#  - hourly


# %%codecell

#no wind label
no_wind_mask = np.logical_and(combined_data_with_nans.WIND_SPEED == 0, combined_data_with_nans.WIND_DIRECTION == 0);

new_combined_data_with_nans = combined_data_with_nans[ no_wind_mask ].assign( NO_WIND=1 ).append( combined_data_with_nans[ np.logical_not( no_wind_mask ) ].assign( NO_WIND=0 )  ).sort_index()

combined_data_with_nans.equals( new_combined_data_with_nans.iloc[:,:-1] )

new_combined_data_with_nans

#weekend label
def is_weekend( dayofweek ):
    return 1 if dayofweek >= 5 else 0;

# 2 – 25 Aug 2013
# https://www.bbc.co.uk/events/rgjc6q/by/date/2013
fringe13 = pd.date_range('2013-08-02 00:00','2013-08-25 23:00', freq='H', tz='GMT')

# 1 – 24 Aug 2014
# https://www.bbc.co.uk/events/rgjc6q/by/date/2014
fringe14 = pd.date_range('2014-08-01 00:00','2014-08-24 23:00', freq='H', tz='GMT')

# 4 – 26 Aug 2015
# https://www.bbc.co.uk/events/rgjc6q/by/date/2015
fringe15 = pd.date_range('2015-08-04 00:00','2015-08-26 23:00', freq='H', tz='GMT')

def is_fringe( timestamp ):
    return 1 if ( timestamp in fringe13 or timestamp in fringe14 or timestamp in fringe15 ) else 0;


# 24-31 December 2013	University closed
# 1-2 January 2014	University closed
# https://www.ed.ac.uk/semester-dates/201314
winter_closure1314 = pd.date_range('2013-12-24 00:00','2014-01-02 23:00', freq='H', tz='GMT')

# 24-31 December 2014	University closed
# 1-2 January 2015	University closed
# https://www.ed.ac.uk/semester-dates/201415
winter_closure1415 = pd.date_range('2014-12-24 00:00','2015-01-02 23:00', freq='H', tz='GMT')

# 24 December 2015 - 4 January 2016	University closed
# https://www.ed.ac.uk/semester-dates/201516
winter_closure1516 = pd.date_range('2015-12-24 00:00','2016-01-04 23:00', freq='H', tz='GMT')

def is_winter_closure( timestamp ):
    return 1 if ( timestamp in winter_closure1314 or timestamp in winter_closure1415 or timestamp in winter_closure1516 ) else 0;

new_combined_data_with_nans = new_combined_data_with_nans.assign(
        WEEKEND=new_combined_data_with_nans.index.to_series().dt.dayofweek.map( is_weekend ).to_numpy(),
        FRINGE=new_combined_data_with_nans.index.to_series().map( is_fringe ).to_numpy(),
        WINTER_CLOSURE=new_combined_data_with_nans.index.to_series().map( is_winter_closure ).to_numpy()
    )

new_combined_data_with_nans_daily = combined_data_with_nans_daily.assign(
        WEEKEND=combined_data_with_nans_daily.index.to_series().dt.dayofweek.map( is_weekend ).to_numpy(),
        FRINGE=combined_data_with_nans_daily.index.to_series().map( is_fringe ).to_numpy(),
        WINTER_CLOSURE=combined_data_with_nans_daily.index.to_series().map( is_winter_closure ).to_numpy()
    )



#holiday label


# %%markdown
# %%codecell
from windrose import WindroseAxes;

combined_data_with_nans['WIND_SPEED'].describe()

fig=plt.figure( figsize=(10,15) )
ax = fig.add_subplot(2, 2, 1, projection="windrose", rmax = 50)
ax.bar( combined_data_with_nans['WIND_DIRECTION'], combined_data_with_nans['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')

ax = fig.add_subplot(2, 2, 2, projection="windrose", rmax = 50)
ax.bar( combined_data_with_nans_daily['WIND_DIRECTION'], combined_data_with_nans_daily['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')

ax = fig.add_subplot(2, 2, 3, projection="windrose", rmax = 50)
ax.bar( combined_data_with_nans_weekly['WIND_DIRECTION'], combined_data_with_nans_weekly['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')

fig.legend(labels=['0.0 - 8.0','8.0 - 16.0','16.0 - 24.0', '24.0 - 32.0', '32.0 - infinity'],   # The labels for each line
           loc="center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Wind Speed Colour Guide"  # Title for the legend
           )

plt.show()

# %%markdown
# %%codecell
combined_data_with_nans_weekly['2016'].index.to_series().dt.weekofyear
from matplotlib.dates import DateFormatter;


foobar = combined_data_with_nans_weekly['2013'];
foobar.index = combined_data_with_nans_weekly['2013'].index.to_series().dt.weekofyear.to_numpy()
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:

    temp = combined_data_with_nans_weekly[ year ];
    temp.index = combined_data_with_nans_weekly[ year ].index.to_series().dt.weekofyear.to_numpy()
    temp = temp.assign( year=year )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()
foobar = foobar.rename( columns={'index':'weekofyear'})

#with sns.axes_style("whitegrid"):
plt.figure(figsize=(15,8))
ax = sns.lineplot( data=foobar, x='weekofyear', y='bin_kwh', err_style='band', ci="sd")
#ax.set_title('Annual bin_kwh 2013-2020 (weekly)')
ax.set_xlabel('Month')
ax.set_ylabel('Electricity Consumption (kWh)')
ax.set_xticks(np.arange(1,53,4+1/3))
ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set(ylim=(0,50000))
plt.show()

# %%markdown
# %%codecell

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar, x='weekofyear', y='bin_kwh', hue='year', legend='full')
    #   ax.set_title('Annual bin_kwh 2013-2020 (weekly)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks(np.arange(1,53,4+1/3))
    ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set(ylim=(0,50000))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()



# %%markdown
# %%codecell


foobar = combined_data_with_nans_daily['2013'];
foobar.index = pd.to_datetime( combined_data_with_nans_daily['2013'].index.strftime('%m-%d-00') )
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:

    temp = combined_data_with_nans_daily[ year ];
    temp.index = pd.to_datetime( combined_data_with_nans_daily[ year ].index.strftime('%m-%d-00') )
    temp = temp.assign( year=year )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()


plt.figure(figsize=(15,8))
ax = sns.lineplot( data=foobar, x='OB_TIME', y='bin_kwh', err_style='band', ci="sd")
ax.xaxis.set_major_formatter( DateFormatter('%m') )
ax.set_title('Annual bin_kwh 2013-2020 (daily)')
ax.set_xlabel('Month')
ax.set_xticks(pd.date_range('2000-01-01','2000-12-31', freq='MS'))
ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.show()

# %%markdown
# %%codecell

fig, axs = plt.subplots(1, 2, figsize=(10,5))
sns.lineplot( data=combined_data_with_nans_daily['2014-03':'2014-05']['bin_kwh'], err_style='band', ci='sd', ax=axs[0])
axs[0].set_xlabel('Month')
axs[0].set_ylabel('Electricity Consumption (kWh)')
axs[0].set_xticks(pd.date_range('2014-03-01','2014-05-31', freq='MS'))
axs[0].set_xticklabels(["Mar",'Apr','May'])
axs[0].set(ylim=(0, 7500))
sns.lineplot( data=combined_data_with_nans_daily['2018-03':'2018-05']['bin_kwh'], err_style='band', ci='sd', ax=axs[1])
axs[1].set_xlabel('Month')
axs[1].set_ylabel('Electricity Consumption (kWh)')
axs[1].set_xticks(pd.date_range('2018-03-01','2018-05-31', freq='MS'))
axs[1].set_xticklabels(["Mar",'Apr','May'])
axs[1].set(ylim=(0, 7500))
plt.tight_layout()
plt.show()

# %%markdown
# %%codecell

spring_mask = np.logical_and( foobar['OB_TIME'] >= '2000-03-01', foobar['OB_TIME'] <= '2000-05-31')


with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar[spring_mask], x='OB_TIME', y='bin_kwh', err_style='band', ci='sd')
    ax.xaxis.set_major_formatter( DateFormatter('%m') )
    #ax.set_title('Spring (Mar - May) bin_kwh 2013-2020')
    ax.set_xlabel('Month')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks(pd.date_range('2000-03-01','2000-05-31', freq='MS'))
    ax.set_xticklabels(["Mar",'Apr','May'])
    ax.set(ylim=(2500, 7000))
    plt.show()

# %%markdown
# %%codecell

summer_mask = np.logical_and( foobar['OB_TIME'] >= '2000-06-01', foobar['OB_TIME'] <= '2000-08-31')

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar[summer_mask], x='OB_TIME', y='bin_kwh', err_style='band', ci="sd")
    ax.xaxis.set_major_formatter( DateFormatter('%m') )
    #ax.set_title('Summer (June - August) bin_kwh 2013-2019')
    ax.set_xlabel('Month')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks(pd.date_range('2000-6','2000-08-31', freq='MS'))
    ax.set_xticklabels(["Jun",'Jul','Aug'])
    ax.set(ylim=(2500, 7000))
    plt.show()

# %%markdown
# %%codecell

autumn_mask = np.logical_and( foobar['OB_TIME'] >= '2000-09-01', foobar['OB_TIME'] <= '2000-11-30')

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar[autumn_mask], x='OB_TIME', y='bin_kwh', err_style='band', ci="sd")
    ax.xaxis.set_major_formatter( DateFormatter('%m') )
    #ax.set_title('Autumn (September - November) bin_kwh 2013-2019')
    ax.set_xlabel('Month')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks(pd.date_range('2000-9','2000-11-30', freq='MS'))
    ax.set_xticklabels(["Sep",'Oct','Nov'])
    ax.set(ylim=(2500, 7000))
    plt.show()

# %%markdown
# %%codecell


foobar = combined_data_with_nans_daily['2013-12-01':'2014-2'];
foobar.index = np.arange( len( combined_data_with_nans_daily['2013-12-01':'2014-2'].index ) )
foobar = foobar.assign( year='2013' )

for year in [2014, 2015, 2016, 2017, 2018, 2019]:
    temp = combined_data_with_nans_daily[ str(year)+'-12-01': str(year+1)+'-2'];
    temp.index = np.arange( len( combined_data_with_nans_daily[str(year)+'-12-01': str(year+1)+'-2'].index ) )
    temp = temp.assign( year=str(year) )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()

foobar = foobar.rename(columns={'index':'day'})

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar, x='day', y='bin_kwh', err_style='band', ci="sd")
    ax.xaxis.set_major_formatter( DateFormatter('%m') )
    #ax.set_title('Winter (December - Feburary) bin_kwh 2013-2020')
    ax.set_xlabel('Month')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_yticks( np.arange(2500, 7500, 500) )
    ax.set_xticks( np.arange(0,91,30) )
    ax.set_xticklabels(["Dec",'Jan','Feb'])
    ax.set(ylim=(2500, 7000))
#    plt.savefig('/Users/lautinyeung/Desktop/foo.pdf',
#            dpi=300,
#            orientation='portrait',
#            bbox_inches='tight', pad_inches=0)
    plt.show()

# %%markdown
# %%codecell

# %%markdown
# %%codecell



foobar = combined_data_with_nans_daily['2013'];
foobar.index = pd.to_datetime( combined_data_with_nans_daily['2013'].index.strftime('%m-%d-00') )
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:

    temp = combined_data_with_nans_daily[ year ];
    temp.index = pd.to_datetime( combined_data_with_nans_daily[ year ].index.strftime('%m-%d-00') )
    temp = temp.assign( year=year )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    spring_mask = np.logical_and( foobar['OB_TIME'] >= '2000-03-01', foobar['OB_TIME'] <= '2000-05-31')

    plt.figure(figsize=(15,8))
    axs[0,0] = sns.lineplot( ax=axs[0,0], data=foobar[spring_mask], x='OB_TIME', y='bin_kwh', err_style='band', ci='sd')
    axs[0,0].xaxis.set_major_formatter( DateFormatter('%m') )
    axs[0,0].set_title('(a) Spring')
    axs[0,0].set_xlabel('')
    axs[0,0].set_ylabel('Electricity Consumption (kWh)')
    axs[0,0].set_xticks(pd.date_range('2000-03-01','2000-05-31', freq='MS'))
    axs[0,0].set_xticklabels(["Mar",'Apr','May'])
    axs[0,0].set(ylim=(0, 7000))




    summer_mask = np.logical_and( foobar['OB_TIME'] >= '2000-06-01', foobar['OB_TIME'] <= '2000-08-31')


    plt.figure(figsize=(15,8))
    axs[0,1] = sns.lineplot( ax=axs[0,1], data=foobar[summer_mask], x='OB_TIME', y='bin_kwh', err_style='band', ci="sd")
    axs[0,1].xaxis.set_major_formatter( DateFormatter('%m') )
    axs[0,1].set_title('(b) Summer')
    axs[0,1].set_xlabel('')
    axs[0,1].set_ylabel('')
    axs[0,1].set_xticks(pd.date_range('2000-6','2000-08-31', freq='MS'))
    axs[0,1].set_xticklabels(["Jun",'Jul','Aug'])
    axs[0,1].set(ylim=(0, 7000))


    autumn_mask = np.logical_and( foobar['OB_TIME'] >= '2000-09-01', foobar['OB_TIME'] <= '2000-11-30')


    plt.figure(figsize=(15,8))
    axs[1,0] = sns.lineplot( ax=axs[1,0], data=foobar[autumn_mask], x='OB_TIME', y='bin_kwh', err_style='band', ci="sd")
    axs[1,0].xaxis.set_major_formatter( DateFormatter('%m') )
    axs[1,0].set_title('(c) Autumn')
    axs[1,0].set_xlabel('Month')
    axs[1,0].set_ylabel('Electricity Consumption (kWh)')
    axs[1,0].set_xticks(pd.date_range('2000-9','2000-11-30', freq='MS'))
    axs[1,0].set_xticklabels(["Sep",'Oct','Nov'])
    axs[1,0].set(ylim=(0, 7000))


    foobar = combined_data_with_nans_daily['2013-12-01':'2014-2'];
    foobar.index = np.arange( len( combined_data_with_nans_daily['2013-12-01':'2014-2'].index ) )
    foobar = foobar.assign( year='2013' )

    for year in [2014, 2015, 2016, 2017, 2018, 2019]:
        temp = combined_data_with_nans_daily[ str(year)+'-12-01': str(year+1)+'-2'];
        temp.index = np.arange( len( combined_data_with_nans_daily[str(year)+'-12-01': str(year+1)+'-2'].index ) )
        temp = temp.assign( year=str(year) )

        foobar = foobar.append( temp )

    foobar = foobar.reset_index()

    foobar = foobar.rename(columns={'index':'day'})


    plt.figure(figsize=(15,8))
    axs[1,1] = sns.lineplot( ax=axs[1,1], data=foobar, x='day', y='bin_kwh', err_style='band', ci="sd")
    axs[1,1].xaxis.set_major_formatter( DateFormatter('%m') )
    axs[1,1].set_title('(d) Winter')
    axs[1,1].set_xlabel('Month')
    axs[1,1].set_ylabel('')
    axs[1,1].set_yticks( np.arange(0, 8000, 1000) )
    axs[1,1].set_xticks( np.arange(0,91,30) )
    axs[1,1].set_xticklabels(["Dec",'Jan','Feb'])
    axs[1,1].set(ylim=(0, 7000))
    #    plt.savefig('/Users/lautinyeung/Desktop/foo.pdf',
    #            dpi=300,
    #            orientation='portrait',
    #            bbox_inches='tight', pad_inches=0)
    plt.show()

# %%markdown
# %%codecell



foobar = new_combined_data_with_nans['2013-03-01':'2013-05-31'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019','2020']:

    temp = new_combined_data_with_nans[year+'-03-01':year+'-05-31'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(2, 2, figsize=(23,10))
    axs[0,0] = sns.lineplot( ax=axs[0,0], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[0,0] = sns.lineplot( ax=axs[0,0], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[0,0].set_title('(a) Spring')
    axs[0,0].set_xlabel(' ')
    axs[0,0].set_ylabel('Electricity Consumption (kWh)')
    axs[0,0].set(ylim=(0, 350))
    axs[0,0].set_xticks( np.arange(0,24) )


    foobar = new_combined_data_with_nans['2013-06-01':'2013-08-31'];
    foobar = foobar.reset_index();
    foobar.index = foobar.OB_TIME.dt.hour;
    foobar = foobar.assign( year='2013' )

    for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

        temp = new_combined_data_with_nans[year+'-06-01':year+'-08-31'];
        temp = temp.reset_index();
        temp.index = temp.OB_TIME.dt.hour;
        temp = temp.assign( year=year )

        foobar = foobar.append(temp)

    foobar.index.name='hourofday'
    foobar = foobar.reset_index()


    axs[0,1] = sns.lineplot( ax=axs[0,1], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[0,1] = sns.lineplot( ax=axs[0,1], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[0,1].set_title('(b) Summer')
    axs[0,1].set_xlabel(' ')
    axs[0,1].set_ylabel('')
    axs[0,1].set_xticks( np.arange(0,24) )
    axs[0,1].set(ylim=(0, 350))


    foobar = new_combined_data_with_nans['2013-09-01':'2013-11-30'];
    foobar = foobar.reset_index();
    foobar.index = foobar.OB_TIME.dt.hour;
    foobar = foobar.assign( year='2013' )

    for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

        temp = new_combined_data_with_nans[year+'-09-01':year+'-11-30'];
        temp = temp.reset_index();
        temp.index = temp.OB_TIME.dt.hour;
        temp = temp.assign( year=year )

        foobar = foobar.append(temp)

    foobar.index.name='hourofday'
    foobar = foobar.reset_index()


    axs[1,0] = sns.lineplot( ax=axs[1,0], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[1,0] = sns.lineplot( ax=axs[1,0], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[1,0].set_title('(c) Autumn')
    axs[1,0].set_xlabel('Hour of Day')
    axs[1,0].set_ylabel('Electricity Consumption (kWh)')
    axs[1,0].set_xticks( np.arange(0,24) )
    axs[1,0].set(ylim=(0, 350))




    foobar = new_combined_data_with_nans['2013-12-01':'2014-2'];
    foobar = foobar.reset_index();
    foobar.index = foobar.OB_TIME.dt.hour;
    foobar = foobar.assign( year='2013' )

    for year in [2014, 2015, 2016, 2017, 2018, 2019, 2020]:

        temp = new_combined_data_with_nans[ str(year)+'-12-01': str(year+1)+'-2' ];
        temp = temp.reset_index();
        temp.index = temp.OB_TIME.dt.hour;
        temp = temp.assign( year=str(year) )

        foobar = foobar.append(temp)

    foobar.index.name='hourofday'
    foobar = foobar.reset_index()


    axs[1,1] = sns.lineplot( ax=axs[1,1], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[1,1] = sns.lineplot( ax=axs[1,1], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    axs[1,1].set_title('(d) Winter')
    axs[1,1].set_xlabel('Hour of Day')
    axs[1,1].set_ylabel('')
    axs[1,1].set_xticks( np.arange(0,24) )
    axs[1,1].set(ylim=(0, 350))
    plt.show()

# %%markdown
# %%codecell

foobar = new_combined_data_with_nans['2013-03-01':'2013-05-31'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019','2020']:

    temp = new_combined_data_with_nans[year+'-03-01':year+'-05-31'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    plt.figure( figsize=(15,8) )
    ax = sns.lineplot( label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    ax = sns.lineplot( label='weekend', ax=ax, data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set(ylim=(125, 350))
    ax.set_xticks( np.arange(0,24) )
    plt.show()

# %%markdown
# %%codecell

foobar = new_combined_data_with_nans['2013-06-01':'2013-08-31'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

    temp = new_combined_data_with_nans[year+'-06-01':year+'-08-31'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    plt.figure( figsize=(15,8) )
    ax = sns.lineplot( label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    ax = sns.lineplot( label='weekend', ax=ax, data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    #ax.set_title('Summer (June - August) hourly bin_kwh 2013-2019')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks( np.arange(0,24) )
    ax.set(ylim=(125, 350))
    plt.show()

# %%markdown
# %%codecell

foobar = new_combined_data_with_nans['2013-09-01':'2013-11-30'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

    temp = new_combined_data_with_nans[year+'-09-01':year+'-11-30'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    ax = sns.lineplot( label='weekend', ax=ax, data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    #ax.set_title('Autumn (September - November) hourly bin_kwh 2013-2019')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks( np.arange(0,24) )
    ax.set(ylim=(125, 350))
    plt.show()

# %%markdown
# %%codecell


foobar = new_combined_data_with_nans['2013-12-01':'2014-2'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in [2014, 2015, 2016, 2017, 2018, 2019, 2020]:

    temp = new_combined_data_with_nans[ str(year)+'-12-01': str(year+1)+'-2' ];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=str(year) )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    ax = sns.lineplot( label='weekend', ax=ax, data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='bin_kwh', err_style='band', ci="sd")
    #ax.set_title('Winter (December - Feburary) hourly bin_kwh 2013-2020')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity Consumption (kWh)')
    ax.set_xticks( np.arange(0,24) )
    ax.set(ylim=(125, 350))
    plt.show()



# %%markdown
# %%codecell

foobar = new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ]['2013-03-01':'2013-05-31'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

    temp = new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ][year+'-03-01':year+'-05-31'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd", label='Spring')
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    ax.set_xlabel('Hour of Day')
    ax.set_xticks( np.arange(0,24) )


foobar = new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ]['2013-06-01':'2013-08-31'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

    temp = new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ][year+'-06-01':year+'-08-31'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

plt.figure(figsize=(15,8))
ax = sns.lineplot( data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd", ax=ax, label='Summer')
#ax.set_title('Summer (June - August) hourly bin_kwh 2013-2019')





foobar = new_combined_data_with_nans['2013-09-01':'2013-11-30'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019']:

    temp = new_combined_data_with_nans[year+'-09-01':year+'-11-30'];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

plt.figure(figsize=(15,8))
ax = sns.lineplot( data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd", ax=ax, label='Autumn')
#ax.set_title('Autumn (September - November) hourly bin_kwh 2013-2019')



foobar = new_combined_data_with_nans['2013-12-01':'2014-2'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in [2014, 2015, 2016, 2017, 2018, 2019]:

    temp = new_combined_data_with_nans[ str(year)+'-12-01': str(year+1)+'-2' ];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=str(year) )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()


plt.figure(figsize=(15,8))
ax = sns.lineplot( data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='bin_kwh', err_style='band', ci="sd", ax=ax, label='Winter')
ax.legend()
#ax.set_title('Winter (December - Feburary) hourly bin_kwh 2013-2020')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Electricity Consumption (kWh)')
ax.set_xticks( np.arange(0,24) )
plt.show()


# %%markdown
# %%codecell

foobar = new_combined_data_with_nans['2013'];
foobar = foobar.reset_index();
foobar.index = foobar.OB_TIME.dt.hour;
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:

    temp = new_combined_data_with_nans[year];
    temp = temp.reset_index();
    temp.index = temp.OB_TIME.dt.hour;
    temp = temp.assign( year=year )

    foobar = foobar.append(temp)

foobar.index.name='hourofday'
foobar = foobar.reset_index()

with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    axs[0] = sns.lineplot( ax=axs[0], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='AIR_TEMPERATURE', err_style='band', ci="sd")
    axs[0] = sns.lineplot( ax=axs[0], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='AIR_TEMPERATURE', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[0].set_xlabel('Hour of Day')
    axs[0].set_ylabel('Temperature (Degree Celcius)')
    axs[0].set(ylim=(0, 20))
    axs[0].set_xticks( np.arange(0,24, 4) )

    #plt.figure( figsize=(15,8) )
    axs[1] = sns.lineplot( ax=axs[1], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='DEWPOINT', err_style='band', ci="sd")
    axs[1] = sns.lineplot( ax=axs[1], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='DEWPOINT', err_style='band', ci="sd")
    axs[1].set_xlabel('Hour of Day')
    axs[1].set_ylabel('')
    axs[1].set(ylim=(0, 20))
    axs[1].set_xticks( np.arange(0,24, 4) )

    #plt.figure( figsize=(15,8) )
    axs[2] = sns.lineplot( ax=axs[2], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='WETB_TEMP', err_style='band', ci="sd")
    axs[2] = sns.lineplot( ax=axs[2], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='WETB_TEMP', err_style='band', ci="sd")
    axs[2].set_xlabel('Hour of Day')
    axs[2].set_ylabel('')
    axs[2].set(ylim=(0, 20))
    axs[2].set_xticks( np.arange(0,24, 4) )

    plt.tight_layout()
    plt.show()

# %%markdown
# %%codecell

with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0] = sns.lineplot( ax=axs[0], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='MSL_PRESSURE', err_style='band', ci="sd")
    axs[0] = sns.lineplot( ax=axs[0], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='MSL_PRESSURE', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[0].set_xlabel('Hour of Day')
    axs[0].set_ylabel('Pressure (hPa)')
    axs[0].set_xticks( np.arange(0,24, 4) )

    axs[1] = sns.lineplot( ax=axs[1], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='STN_PRES', err_style='band', ci="sd")
    axs[1] = sns.lineplot( ax=axs[1], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='STN_PRES', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[1].set_xlabel('Hour of Day')
    axs[1].set_ylabel('')
    #ax.set(ylim=(0, 16))
    axs[1].set_xticks( np.arange(0,24, 4) )


    plt.tight_layout()
    plt.show()

# %%markdown
# %%codecell
new_combined_data_with_nans.shape
new_combined_data_with_nans[new_combined_data_with_nans['WMO_HR_SUN_DUR'].isna()].shape
foobar[ np.logical_and(foobar['hourofday'] == 0, np.logical_or(foobar['WMO_HR_SUN_DUR'] == 1, foobar['WMO_HR_SUN_DUR'] == 0))]

with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0] = sns.lineplot( ax=axs[0], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='DRV_HR_SUN_DUR', err_style='band', ci="sd")
    axs[0] = sns.lineplot( ax=axs[0], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='DRV_HR_SUN_DUR', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[0].set_xlabel('Hour of Day')
    axs[0].set_ylabel('Duration of sunshine (hours)')
    axs[0].set_xticks( np.arange(0,24, 4) )

    axs[1] = sns.lineplot( ax=axs[1], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='WMO_HR_SUN_DUR', err_style='band', ci="sd")
    axs[1] = sns.lineplot( ax=axs[1], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='WMO_HR_SUN_DUR', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[1].set_xlabel('Hour of Day')
    axs[1].set_ylabel('')
    #ax.set(ylim=(0, 16))
    axs[1].set_xticks( np.arange(0,24, 4) )


    plt.tight_layout()
    plt.show()

# %%markdown
# %%codecell

foobar[np.logical_and(foobar['hourofday'] == 21, foobar['PRCP_AMT'] == 0)]
with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(2, 2, figsize=(15,10))

    i=0
    ylabels = ['Relative Humidity (%)', 'Precipitation amount (mm)', 'Cloud-base height (decametres)', 'Visibility (decametres)']
    for feature in ['RLTV_HUM','PRCP_AMT','CLD_BASE_HT','VISIBILITY']:

        sns.lineplot( ax=axs[int(i/2),i%2], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y=feature, err_style='band', ci="sd")
        sns.lineplot( ax=axs[int(i/2),i%2], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y=feature, err_style='band', ci="sd")
        axs[int(i/2),i%2].set_title(feature)
        axs[int(i/2),i%2].set_xlabel('Hour of Day')
        axs[int(i/2),i%2].set_ylabel(ylabels[i])
        axs[int(i/2),i%2].set_xticks( np.arange(0,24, 4) )

        i=i+1
    plt.tight_layout()
    plt.show()


# %%markdown
# %%codecell

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='WIND_SPEED', err_style='band', ci="sd")
    sns.lineplot( ax=ax, label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='WIND_SPEED', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Windspeed (knots)')
    ax.set_xticks( np.arange(0,24, 1) )


# %%markdown
# %%codecell


with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0] = sns.lineplot( ax=axs[0], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='RLTV_HUM', err_style='band', ci="sd")
    axs[0] = sns.lineplot( ax=axs[0], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='RLTV_HUM', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[0].set_xlabel('Hour of Day')
    axs[0].set_ylabel('Relative Humidity (%)')
    axs[0].set_xticks( np.arange(0,24, 4) )

    axs[1] = sns.lineplot( ax=axs[1], label='weekday', data=foobar[foobar['WEEKEND'] == 0], x='hourofday', y='WMO_HR_SUN_DUR', err_style='band', ci="sd")
    axs[1] = sns.lineplot( ax=axs[1], label='weekend', data=foobar[foobar['WEEKEND'] == 1], x='hourofday', y='WMO_HR_SUN_DUR', err_style='band', ci="sd")
    #ax.set_title('Spring (March - May) hourly bin_kwh 2013-2020')
    axs[1].set_xlabel('Hour of Day')
    axs[1].set_ylabel('Relative Humidity (%)')
    #ax.set(ylim=(0, 16))
    axs[1].set_xticks( np.arange(0,24, 4) )


    plt.tight_layout()
    plt.show()

# %%markdown
# %%codecell

foobar = combined_data_with_nans_weekly['2013'];
foobar.index = combined_data_with_nans_weekly['2013'].index.to_series().dt.weekofyear.to_numpy()
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:

    temp = combined_data_with_nans_weekly[ year ];
    temp.index = combined_data_with_nans_weekly[ year ].index.to_series().dt.weekofyear.to_numpy()
    temp = temp.assign( year=year )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()
foobar = foobar.rename( columns={'index':'weekofyear'})

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar, x='weekofyear', y='AIR_TEMPERATURE', label='AIR_TEMPERATURE', err_style='band', ci="sd")
    sns.lineplot( ax=ax, data=foobar, x='weekofyear', y='DEWPOINT', label='DEWPOINT', err_style='band', ci="sd")
    sns.lineplot( ax=ax, data=foobar, x='weekofyear', y='WETB_TEMP', label='WETB_TEMP', err_style='band', ci="sd")
    ax.legend()
    ax.set_xticks(np.arange(1,53,4+1/3))
    ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Temperature (Degree Celcius)')
    plt.show()

# %%markdown
# %%codecell

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar, x='weekofyear', y='MSL_PRESSURE', label='MSL_PRESSURE', err_style='band', ci="sd")
    sns.lineplot( ax=ax, data=foobar, x='weekofyear', y='STN_PRES', label='STN_PRES', err_style='band', ci="sd")
    ax.legend()
    ax.set_xticks(np.arange(1,53,4+1/3))
    ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Pressure (hPa)')
    plt.show()

# %%markdown
# %%codecell

with sns.axes_style("whitegrid"):
    plt.figure(figsize=(15,8))
    ax = sns.lineplot( data=foobar, x='weekofyear', y='WMO_HR_SUN_DUR', label='WMO_HR_SUN_DUR', err_style='band', ci="sd")
    sns.lineplot( ax=ax, data=foobar, x='weekofyear', y='DRV_HR_SUN_DUR', label='DRV_HR_SUN_DUR', err_style='band', ci="sd")
    ax.legend()
    ax.set_xticks(np.arange(1,53,4+1/3))
    ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Duration of sunshine (hours)')
    plt.show()

# %%markdown
# %%codecell

#with sns.axes_style("whitegrid"):
fig, axs = plt.subplots(2, 2, figsize=(15,10))

i=0
ylabels = ['Relative Humidity (%)', 'Precipitation amount (mm)', 'Cloud-base height (decametres)', 'Visibility (decametres)']
for feature in ['RLTV_HUM','PRCP_AMT','CLD_BASE_HT','VISIBILITY']:

    sns.lineplot( ax=axs[int(i/2),i%2], data=foobar, x='weekofyear', y=feature, err_style='band', ci="sd")
    axs[int(i/2),i%2].set_title(feature)
    axs[int(i/2),i%2].set_xlabel('Month')
    axs[int(i/2),i%2].set_ylabel(ylabels[i])
    axs[int(i/2),i%2].set_xticks(np.arange(1,53,8+2/3))
    axs[int(i/2),i%2].set_xticklabels(['Jan',"Mar",'May','Jul','Sep','Nov'])

    i=i+1
plt.tight_layout()
plt.show()


# %%markdown
# %%codecell

fig, axs = plt.subplots(4, 4, figsize=(15,15))

for i in range(13):
    feature = combined_data_with_nans.columns[i];

    sns.lineplot( ax=axs[int(i/4),i%4], data=foobar, x='weekofyear', y=feature, err_style='band', ci="sd")
    axs[int(i/4),i%4].set_title(feature)
    axs[int(i/4),i%4].set_xlabel('Month')
    axs[int(i/4),i%4].set_xticks(np.arange(1,53,8+2/3))
    axs[int(i/4),i%4].set_xticklabels(['Jan',"Mar",'May','Jul','Sep','Nov'])

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(3, 4, figsize=(15,15))

for i in range(12):
    feature = combined_data_with_nans.columns[i];

    ax = sns.lineplot( ax=axs[int(i/4),i%4], data=foobar[foobar['year'] != '2020'], x='weekofyear', y=feature, hue='year', legend=False)
    axs[int(i/4),i%4].set_title(feature)
    axs[int(i/4),i%4].set_xlabel('Month')
    axs[int(i/4),i%4].set_xticks(np.arange(1,53,8+2/3))
    axs[int(i/4),i%4].set_xticklabels(['Jan',"Mar",'May','Jul','Sep','Nov'])


fig.legend(labels=['2013','2014', '2015', '2016', '2017', '2018', '2019'],   # The labels for each line
           loc="center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Year"  # Title for the legend
           )
plt.tight_layout()
plt.show()


# %%markdown
# %%codecell


ax = WindroseAxes.from_ax()
ax.bar( new_combined_data_with_nans_weekly['WIND_DIRECTION'], new_combined_data_with_nans_weekly['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')
ax.set_title('Annual')
ax.legend(labels=['[0.0 - 8.0)','[8.0 - 16.0)','[16.0 - 24.0)', '[24.0 - 32.0)', '[32.0 - infinity)'],   # The labels for each line
           loc="lower right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Wind Speed Colour Guide"  # Title for the legend
           )

plt.tight_layout()

# %%markdown
# %%codecell

foobar = combined_data_with_nans_weekly['2013'];
foobar.index = pd.to_datetime( combined_data_with_nans_weekly['2013'].index.strftime('%m-%d-00') )
foobar = foobar.assign( year='2013' )

for year in ['2014', '2015', '2016', '2017', '2018', '2019', '2020']:

    temp = combined_data_with_nans_weekly[ year ];
    temp.index = pd.to_datetime( combined_data_with_nans_weekly[ year ].index.strftime('%m-%d-00') )
    temp = temp.assign( year=year )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()

fig=plt.figure( figsize=(20,30) )

ax = fig.add_subplot(2, 2, 1, projection="windrose", rmax = 50)
ax.bar( foobar[ spring_mask ]['WIND_DIRECTION'], foobar[ spring_mask ]['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')
ax.set_title('Spring')



ax = fig.add_subplot(2, 2, 2, projection="windrose", rmax = 50)
ax.bar( foobar[ summer_mask ]['WIND_DIRECTION'], foobar[ summer_mask ]['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')
ax.set_title('Summer')


ax = fig.add_subplot(2, 2, 3, projection="windrose", rmax = 50)
ax.bar( foobar[ autumn_mask ]['WIND_DIRECTION'], foobar[ autumn_mask ]['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')
ax.set_title('Autumn')


foobar = combined_data_with_nans_weekly['2013-12-01':'2014-2'];
foobar.index = np.arange( len( combined_data_with_nans_weekly['2013-12-01':'2014-2'].index ) )
foobar = foobar.assign( year='2013' )

for year in [2014, 2015, 2016, 2017, 2018, 2019, 2020]:
    temp = combined_data_with_nans_weekly[ str(year)+'-12-01': str(year+1)+'-2'];
    temp.index = np.arange( len( combined_data_with_nans_weekly[str(year)+'-12-01': str(year+1)+'-2'].index ) )
    temp = temp.assign( year=str(year) )

    foobar = foobar.append( temp )

foobar = foobar.reset_index()

foobar = foobar.rename(columns={'index':'day'})


ax = fig.add_subplot(2, 2, 4, projection="windrose", rmax = 50)
ax.bar( foobar['WIND_DIRECTION'], foobar['WIND_SPEED'], normed=True, bins=np.arange(0,37,8), nsector=8, opening=0.8, edgecolor='white')
ax.set_title('Winter')

fig.legend(labels=['[0.0 - 8.0)','[8.0 - 16.0)','[16.0 - 24.0)', '[24.0 - 32.0)', '[32.0 - infinity)'],   # The labels for each line
           loc="center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Wind Speed Colour Guide"  # Title for the legend
           )

plt.tight_layout()
plt.show()





# %%markdown
# # HDD CDD
# %%codecell

fix, axs = plt.subplots( 1,2, figsize=(20,10))
plt.figure(figsize=(15,12)) #10,15
sns.heatmap(
    combined_data_with_nans.iloc[:,:-1].corr(method='pearson'),
    cmap=plt.cm.coolwarm,
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.1,
    linecolor='white',
    square=True,
    annot=True,
    #ax=axs[0]
    annot_kws={"size": 15}
)

axs[0].set_title('Hourly')


#plt.figure(figsize=(20,15))
plt.figure(figsize=(15,12))
sns.heatmap(
    combined_data_with_nans_weekly.iloc[:,:-1].corr(method='pearson'),
    cmap=plt.cm.coolwarm,
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.1,
    linecolor='white',
    square=True,
    annot=True,
    #ax=axs[1],
    annot_kws={"size": 15}
)
axs[1].set_title('Weekly')

plt.show()


def to_DD( data ):
    temp = data['AIR_TEMPERATURE'];

    # arbritary, follow literatur.  Can base on summary stat, BUT BECAREFUL of data leak ( train set vs. test set)
    hdd_base = 15.5;
    #hdd_base = 20;
    cdd_base = 20;

    hdd = np.maximum( hdd_base - temp, np.zeros_like(temp)).sum()
    cdd = np.maximum( temp - cdd_base, np.zeros_like(temp)).sum()

    return pd.Series([hdd,cdd], index=['WHDH','WCDH'])

dd = combined_data_with_nans[ ['AIR_TEMPERATURE'] ].groupby( pd.Grouper(freq='1w') ).apply( to_DD )

new_combined_data_with_nans_weekly = pd.concat( [dd.iloc[1:], combined_data_with_nans_weekly], axis=1 )

new_combined_data_with_nans_weekly = new_combined_data_with_nans_weekly.assign(
        FRINGE=new_combined_data_with_nans_weekly.index.to_series().map( is_fringe ).to_numpy(),
        WINTER_CLOSURE=new_combined_data_with_nans_weekly.index.to_series().map( is_winter_closure ).to_numpy()
    )

new_combined_data_with_nans_weekly


plt.figure(figsize=(15,12))
sns.heatmap(
    new_combined_data_with_nans_weekly.drop(['bin_kwh','WINTER_CLOSURE','FRINGE'], axis=1).corr(method='kendall'),
    cmap=plt.cm.coolwarm,
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.1,
    linecolor='white',
    square=True,
    annot=True,
    annot_kws={"size": 15}
)
plt.show()



# %% markdown
# %% codecell

import rpy2;
%load_ext rpy2.ipython

import rpy2.robjects as ro;

from rpy2.robjects import pandas2ri;
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter;

from rpy2.robjects.packages import importr;

utils = importr("utils");
grdevices = importr('grDevices');
infenergy = importr('infenergy');


def help_r( python_name ):
    print( str( utils.help( python_name.replace('_','.') ) ) );

combined_data_with_nans.index[0]
combined_data_with_nans.index[-1]



ro.r('''
from <- "2011-07-06"
to <- "2020-06-01"

## Server & Forum UPS data
server1 <- get.ups.hourly(from, "2017-07-16", upss=c("serverL", "serverR"), power.factor=NA, infer.missing.data=FALSE)
server2 <- get.ups.hourly("2017-07-16", "2018-02-12", upss=c("serverL", "serverR"), power.factor=NA, infer.missing.data=TRUE)
server3 <- get.ups.hourly("2018-02-12", to, upss=c("serverL", "serverR"), power.factor=NA, infer.missing.data=FALSE)
server <- rbind(server1, server2, server3)
attr(server, "to") <- as.POSIXlt(to, tz="GMT")

## Separate out bits where we know only forumB was working - otherwise
## the system tries to compensate for the missing forumA
## forum <- get.ups.hourly(from, to, upss=c("forumA", "forumB"),  power.factor=0.9)
forum1 <-  get.ups.hourly(from, "2013-01-03", upss=c("forumA", "forumB"), power.factor=0.9)
forum2 <-  get.ups.hourly("2013-01-03", "2013-03-18", upss=c("forumB"), power.factor=0.9)
forum3 <-  get.ups.hourly("2013-03-18", "2013-03-21", upss=c("forumA", "forumB"), power.factor=0.9)
forum4 <-  get.ups.hourly("2013-03-21", to, upss=c("forumB"), power.factor=0.9)
forum <- rbind(forum1, forum2, forum3, forum4)
attr(forum, "to") <- as.POSIXlt(to, tz="GMT")
''')

ro.r('''
## Read in the main meter data: the first part comes from the
## university; the second from meter readings
## dates *inclusive* dates
meter <- hourly(get.inf.meter.data("2013-01-21", to))
dat <- rbind(meter)
attr(dat, "to") <- as.POSIXlt(to, tz="GMT")

## Set missing server readings to zero
server$kWh[is.na(server$kWh)] <-0
forum$kWh[is.na(forum$kWh)] <-0

combine <- combine.data.hourly(list(Server=server, Forum=forum), tot=dat)


par(mar=c(2,4,1,0))
combine.weekly <- weekly(combine) # THE DATA FRAME
''')



data_ups_combine        = ro.r['combine'] # as an r dataframe
data_ups_combine_weekly = ro.r['combine.weekly'] # as an r dataframe

with localconverter(ro.default_converter + pandas2ri.converter):
  data_ups_combine_converted = ro.conversion.rpy2py( data_ups_combine );
  data_ups_combine_weekly_converted = ro.conversion.rpy2py( data_ups_combine_weekly );

data_ups_combine_converted = data_ups_combine_converted.set_index('Time')
data_ups_combine_weekly_converted = data_ups_combine_weekly_converted.set_index('Time')

data_ups_combine_converted_filtered = data_ups_combine_converted[\
    combined_data_with_nans.index[0].strftime("%m/%d/%Y, %H:%M:%S"):\
        combined_data_with_nans.index[-1].strftime("%m/%d/%Y, %H:%M:%S")
        ]


data_ups_combine_converted_filtered.sum( axis = 1 ).resample('W').sum().plot()

new_combined_data_with_nans['bin_kwh'].resample('W').sum().plot()


data_ups_combine_converted_filtered.describe()
data_ups_combine_converted_filtered.sum( axis = 1 ).describe()


data_ups_combine_converted_filtered.resample('W').sum().plot()
data_ups_combine_converted_filtered.sum( axis = 1 ).resample('W').sum().plot()

combined_data_with_nans['bin_kwh']['2014':'2018'].resample('W').sum().plot()
data_ups_combine_converted_filtered.sum( axis=1 )['2014':'2018'].resample('W').sum().plot()
data_ups_combine_converted_filtered['Forum']['2014-01-20'].plot()

# %% markdown
# MODELLING
# %% codecell
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# feature selection
# def select_features(X_train, y_train, X_test):
# 	fs = SelectKBest(score_func=f_regression, k=6)
# 	fs.fit(X_train, y_train)
# 	# auto select features ( transform )
# 	X_train_fs = fs.transform(X_train)
# 	X_test_fs = fs.transform(X_test)
# 	return X_train_fs, X_test_fs, fs


# weekend_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','bin_kwh']
# weekday_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','DRV_HR_SUN_DUR','bin_kwh']

more_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','DRV_HR_SUN_DUR','PRCP_AMT','bin_kwh']

more_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','DRV_HR_SUN_DUR','PRCP_AMT','bin_kwh']

new_combined_data_with_nans.columns

# X, y = new_combined_data_with_nans_weekly.iloc[:,:-1].assign(week=np.arange(len(new_combined_data_with_nans_weekly))).dropna(), new_combined_data_with_nans_weekly.dropna()['bin_kwh']
X, y = new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ][weekday_features].dropna().iloc[:,:-1], new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ][weekday_features].dropna()['bin_kwh']
#X, y = new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ][weekday_features].dropna().iloc[1:,:-1], new_combined_data_with_nans[ new_combined_data_with_nans['WEEKEND'] == 0 ][weekday_features].dropna()['bin_kwh'].diff().iloc[1:]

#X, y = X[['WMO_HR_SUN_DUR','RLTV_HUM']], y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
# feature selection
#X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


#for i in fs.scores_.argsort()[::-1]:
#	print(f'{new_combined_data_with_nans_weekly.iloc[:,:-1].assign(week=np.arange(len(new_combined_data_with_nans_weekly))).columns[i]}: {fs.scores_[i]}')
# plot the scores
#plt.bar([i for i in fs.scores_.argsort()[::-1]], fs.scores_[fs.scores_.argsort()[::-1]])
#plt.show()

# %% markdown
# %% codecell



X_train.columns
#sns.scatterplot(x=X_train['PRCP_AMT'], y=y_train)
sns.scatterplot(x=X_train['WIND_SPEED'], y=y_train)
sns.scatterplot(x=X_train['AIR_TEMPERATURE'], y=y_train)
sns.scatterplot(x=X_train['RLTV_HUM'], y=y_train)
#sns.scatterplot(x=X_train['MSL_PRESSURE'], y=y_train)
sns.scatterplot(x=X_train['DRV_HR_SUN_DUR'], y=y_train)
sns.scatterplot(x=X_train['VISIBILITY'], y=y_train)
sns.scatterplot(x=X_train['CLD_BASE_HT'], y=y_train)



# %%markdown
# %%codecell

# fit the model
model = LinearRegression()
#model.fit(X_train_fs, y_train)
#model.fit(X_train, y_train)

# evaluate the model
scoring = {'RMSE': make_scorer( lambda y_true, y_pred: mean_squared_error(y_true, y_pred)**0.5 ),
           'MAPE': make_scorer( mean_absolute_percentage_error ),
           'Explained Variance': make_scorer( explained_variance_score ),
           'R2': make_scorer( r2_score )}
scores = cross_validate(model, X, y, scoring=scoring,
                         cv=5, return_train_score=True)

scores

print('Train scores: \n\n'+\
'RMSE: {} += {}\n'.format(scores['train_RMSE'].mean(), scores['train_RMSE'].std())+\
'MAPE: {} , {}\n'.format(scores['train_MAPE'].mean(), scores['train_MAPE'].std())+\
'Explained Variance: {} , {}\n'.format(scores['train_Explained Variance'].mean(), scores['train_Explained Variance'].std())+\
'R2: {} , {}\n'.format(scores['train_R2'].mean(), scores['train_R2'].std())
)

print('Test scores: \n\n'+\
'RMSE: {} += {}\n'.format(scores['test_RMSE'].mean(), scores['test_RMSE'].std())+\
'MAPE: {} , {}\n'.format(scores['test_MAPE'].mean(), scores['test_MAPE'].std())+\
'Explained Variance: {} , {}\n'.format(scores['test_Explained Variance'].mean(), scores['test_Explained Variance'].std())+\
'R2: {} , {}\n'.format(scores['test_R2'].mean(), scores['test_R2'].std())
)

plot_model = LinearRegression();
plot_model.fit(X_train, y_train)

print('Plot model')
#yhat = model.predict(X_test_fs)
yhat = plot_model.predict(X_test)
# evaluate predictions
rmse = mean_squared_error(y_test, yhat)**0.5
mape = mean_absolute_percentage_error(y_test, yhat)
expl_var = explained_variance_score(y_test, yhat)
r2 = r2_score(y_test, yhat)
print('RMSE: %.3f' % rmse)
print('MAPE: %.3f percent' % mape)

#print('RMSE: %.3f' % rmse)
#print('MAPE: %.3f percent' % mape)

#print('RMSE: %.3f' % rmse)
#print('MAPE: %.3f percent' % mape)

print('Explained Variance: %.3f' % expl_var)
print('R2: %.3f' % r2)

#print('Explained Variance: %.3f' % expl_var)
#print('R2: %.3f' % r2)

# print('Explained Variance: %.3f' % expl_var)
# print('R2: %.3f' % r2)

#ax=sns.lineplot(x=X_test.index, y=model.predict(X_test_fs))
foobar = pd.DataFrame(plot_model.predict(X_test))
foobar.index = X_test.index

X_train.shape
X_test.shape

X_train.columns
plot_model.coef_
X_train.columns[plot_model.coef_.argsort()]

#model.coef_

#model.coef_

ax=foobar.resample('W').mean().plot()
y_test.resample('W').mean().plot(ax=ax)

#ax=foobar.resample('W').mean().plot()
#y_test.resample('W').mean().plot(ax=ax)

# ax=foobar.resample('W').mean().plot()
# y_test.resample('W').mean().plot(ax=ax)

#%% markdown
#%% codecell

from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(X.reset_index(), y,
                                                     column_id='OB_TIME')


features_filtered_direct


my_model = LinearRegression()
my_model.fit(features_filtered_direct, y)


# evaluate the model
yhat = my_model.predict(features_filtered_direct)
# evaluate predictions
rmse = mean_squared_error(y, yhat)**0.5
print('RMSE: %.3f' % rmse)

ax=sns.lineplot(x=features_filtered_direct.index, y=my_model.predict(features_filtered_direct))

y.plot(ax=ax)
new_combined_data_with_nans_daily.shape
# %%markdown
# %%codecell
[ feature for feature in combined_data_with_nans if feature not in hourly_features]
# weekend_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','bin_kwh']
# weekday_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','DRV_HR_SUN_DUR','bin_kwh']

hourly_features = ['WIND_DIRECTION', 'WIND_SPEED', 'CLD_BASE_HT', 'VISIBILITY', 'STN_PRES', 'AIR_TEMPERATURE', 'RLTV_HUM', 'DRV_HR_SUN_DUR', 'PRCP_AMT', 'NO_WIND', 'WEEKEND', 'FRINGE', 'WINTER_CLOSURE', 'bin_kwh']
daily_features  = ['WIND_DIRECTION', 'WIND_SPEED', 'CLD_BASE_HT', 'VISIBILITY', 'STN_PRES', 'AIR_TEMPERATURE', 'RLTV_HUM', 'DRV_HR_SUN_DUR', 'PRCP_AMT', 'WEEKEND', 'FRINGE', 'WINTER_CLOSURE', 'bin_kwh']
weekly_features = ['WIND_DIRECTION', 'WIND_SPEED', 'CLD_BASE_HT', 'VISIBILITY', 'STN_PRES', 'WHDH', 'WCDH', 'RLTV_HUM', 'DRV_HR_SUN_DUR', 'PRCP_AMT', 'FRINGE', 'WINTER_CLOSURE', 'bin_kwh']
time_features = [ 'WEEKEND', 'FRINGE', 'WINTER_CLOSURE','bin_kwh']
# more_features = ['AIR_TEMPERATURE', 'WIND_SPEED', 'VISIBILITY', 'RLTV_HUM', 'CLD_BASE_HT','DRV_HR_SUN_DUR','PRCP_AMT','bin_kwh']

using_datas = []

X, y = new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 0 ][hourly_features].dropna().iloc[:,:-1], new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 0 ][hourly_features].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('weekdays 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])

X, y = new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 1 ][hourly_features].dropna().iloc[:,:-1], new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 1 ][hourly_features].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('weekends 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])

# X, y = new_combined_data_with_nans.drop('WIND_DIRECTION', axis=1).dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans.drop('WIND_DIRECTION', axis=1).dropna()['bin_kwh']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

X.shape
X['NO_WIND'].sum()
# X, y = new_combined_data_with_nans[more_features].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[more_features].dropna()['bin_kwh']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# X, y = new_combined_data_with_nans[['DRV_HR_SUN_DUR', 'AIR_TEMPERATURE']].dropna(), new_combined_data_with_nans[['DRV_HR_SUN_DUR', 'AIR_TEMPERATURE','bin_kwh']].dropna()['bin_kwh']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# for kk in using_datas:
#     name, X, y, X_train, X_test, y_train, y_test = kk
#     print( X.shape )

using_datas.extend([('hourly 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])


#X, y = new_combined_data_with_nans_weekly.iloc[:,:-1].assign(week=np.arange(len(new_combined_data_with_nans_weekly))).dropna(), new_combined_data_with_nans_weekly.dropna()['bin_kwh']
X, y = new_combined_data_with_nans_weekly[weekly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_weekly[weekly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('weekly 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])


X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('daily 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])

# %%markdown
# %%codecell

using_datas = []

# X, y = new_combined_data_with_nans.dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans.dropna().join( data_ups_combine_converted_filtered['Other'] )['Other']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#
# using_datas.extend([('ups all', X, y, X_train, X_test, y_train, y_test)])

X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('hourly 2013-02-01 to 2016-01-31 ', X, y, X_train, X_test, y_train, y_test)])

X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('daily 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])


X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().join( data_ups_combine_converted_filtered['Other'] )['Other']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('Removing UPS hourly 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])

X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().join( data_ups_combine_converted_filtered['Other'].resample('D').sum() )['Other']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

using_datas.extend([('Removing UPS daily 2013-02-01 to 2016-01-31', X, y, X_train, X_test, y_train, y_test)])

# %%markdown
# %%codecell
fig, axs = plt.subplots(2, 3, figsize=(25,8))
i=0
for dataset in using_datas:

    name, X, y, X_train, X_test, y_train, y_test = dataset

    print( name )

    model = LinearRegression()

    # evaluate the model
    scoring = {'RMSE': make_scorer( lambda y_true, y_pred: mean_squared_error(y_true, y_pred)**0.5 ),
               'MAPE': make_scorer( mean_absolute_percentage_error ),
               'Explained Variance': make_scorer( explained_variance_score ),
               'R2': make_scorer( r2_score )}
    shuffler = ShuffleSplit(n_splits=10, test_size=0.33, random_state=1)
    scores = cross_validate(model, X, y, scoring=scoring,
                             cv=shuffler, return_train_score=True)

    print('Train scores: \n\n'+\
    'RMSE: {} += {}\n'.format(scores['train_RMSE'].mean(), scores['train_RMSE'].std())+\
    'MAPE: {} += {}\n'.format(scores['train_MAPE'].mean(), scores['train_MAPE'].std())+\
    'Explained Variance: {} += {}\n'.format(scores['train_Explained Variance'].mean(), scores['train_Explained Variance'].std())+\
    'R2: {} += {}\n'.format(scores['train_R2'].mean(), scores['train_R2'].std())
    )

    print('Test scores: \n\n'+\
    'RMSE: {} += {}\n'.format(scores['test_RMSE'].mean(), scores['test_RMSE'].std())+\
    'MAPE: {} += {}\n'.format(scores['test_MAPE'].mean(), scores['test_MAPE'].std())+\
    'Explained Variance: {} += {}\n'.format(scores['test_Explained Variance'].mean(), scores['test_Explained Variance'].std())+\
    'R2: {} += {}\n'.format(scores['test_R2'].mean(), scores['test_R2'].std())
    )

    plot_model = LinearRegression();
    plot_model.fit(X_train, y_train)


    print('Plot model')

    yhat = plot_model.predict(X_test)

    # evaluate metrics
    rmse = mean_squared_error(y_test, yhat)**0.5
    mape = mean_absolute_percentage_error(y_test, yhat)
    expl_var = explained_variance_score(y_test, yhat)
    r2 = r2_score(y_test, yhat)

    print('RMSE: %.3f' % rmse)
    print('MAPE: %.3f percent' % mape)
    print('Explained Variance: %.3f' % expl_var)
    print('R2: %.3f' % r2)
    print('')
    # foobar = pd.DataFrame(plot_model.predict(X_test))
    # foobar.index = X_test.index

    foobar = pd.DataFrame(plot_model.predict(X))
    foobar.index = X.index

    # ax=sns.lineplot(data=foobar.resample('W').mean(), x=foobar.resample('W').mean().index, y=0, label='predicted')
    # ax=sns.lineplot(data=y.resample('W').mean(), label='actual', ax=ax)
    # ax.set_title(name)
    # ax.set_xlabel('Year')
    # ax.set_ylabel('Electricity Consumption (kWh)')
    # plt.show()

    # combined_data_with_nans_daily['bin_kwh']['2013-02':'2016-01-31'].plot()
    # sns.lineplot( data=foobar, x=foobar.index, y=0, label='predicted')

    axs[int(i/3),i%3]=sns.lineplot(ax=axs[int(i/3),i%3], data=foobar.resample('W').sum(), x=foobar.resample('W').sum().index, y=0, label='predicted')
    axs[int(i/3),i%3]=sns.lineplot(data=y.resample('W').sum(), label='actual', ax=axs[int(i/3),i%3])
    axs[int(i/3),i%3].set_title(name)
    axs[int(i/3),i%3].set_xlabel('')
    axs[int(i/3),i%3].set_ylabel('')
    axs[int(i/3),i%3].set(ylim=(-1000,45000))
    axs[int(i/3),i%3].set_xticks(pd.date_range('2013-01','2016-01-31', freq='AS'))

    i=i+1
axs[0,0].set_ylabel('Electricity Consumption (kWh)')
axs[1,0].set_ylabel('Electricity Consumption (kWh)')
axs[1,0].set_xlabel('Year')
axs[1,1].set_xlabel('Year')
axs[1,2].set_xlabel('Year')
axs[-1,-1].axis('off')
# ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.tight_layout()
plt.show()

    #
    # X_train.shape
    # X_test.shape
    #
    # X_train.columns
    #
    # plot_model.coef_

    # ax=foobar.resample('W').mean().plot()
    # y_test.resample('W').mean().plot(ax=ax)

# %%markdown
# %%codecell
fig, axs = plt.subplots(2, 3, figsize=(25,8))
i=0
for dataset in using_datas:

    name, X, y, X_train, X_test, y_train, y_test = dataset



    # evaluate the model
    scoring = {'RMSE': make_scorer( lambda y_true, y_pred: mean_squared_error(y_true, y_pred)**0.5 ),
               'MAPE': make_scorer( mean_absolute_percentage_error ),
               'Explained Variance': make_scorer( explained_variance_score ),
               'R2': make_scorer( r2_score )}

    foo = GridSearchCV(estimator=Lasso(), scoring=make_scorer(r2_score), param_grid={'alpha': np.arange(0,2,0.05)})
    foo.fit(X_train,y_train);
    alp = foo.best_params_['alpha']
    model = Lasso(alpha=alp)

    print( f'{name}, lambda(alpha)={alp}' )

    shuffler = ShuffleSplit(n_splits=10, test_size=0.33, random_state=1)
    scores = cross_validate(model, X, y, scoring=scoring,
                             cv=shuffler, return_train_score=True)


    print('Train scores: \n\n'+\
    'RMSE: {} += {}\n'.format(scores['train_RMSE'].mean(), scores['train_RMSE'].std())+\
    'MAPE: {} += {}\n'.format(scores['train_MAPE'].mean(), scores['train_MAPE'].std())+\
    'Explained Variance: {} += {}\n'.format(scores['train_Explained Variance'].mean(), scores['train_Explained Variance'].std())+\
    'R2: {} += {}\n'.format(scores['train_R2'].mean(), scores['train_R2'].std())
    )

    print('Test scores: \n\n'+\
    'RMSE: {} += {}\n'.format(scores['test_RMSE'].mean(), scores['test_RMSE'].std())+\
    'MAPE: {} += {}\n'.format(scores['test_MAPE'].mean(), scores['test_MAPE'].std())+\
    'Explained Variance: {} += {}\n'.format(scores['test_Explained Variance'].mean(), scores['test_Explained Variance'].std())+\
    'R2: {} += {}\n'.format(scores['test_R2'].mean(), scores['test_R2'].std())
    )

    plot_model = Lasso(alpha=alp);
    plot_model.fit(X_train, y_train)
    # yhat
    # y_test.var()
    # 1 - (y_test-yhat).var()/y_test.var()
    # 1 - (y_test - np.zeros_like(yhat) + y_test.mean()).var() / y_test.var()
    print('Plot model')

    yhat = plot_model.predict(X_test)

    # evaluate metrics
    rmse = mean_squared_error(y_test, yhat)**0.5
    mape = mean_absolute_percentage_error(y_test, yhat)
    expl_var = explained_variance_score(y_test, yhat)
    r2 = r2_score(y_test, yhat)

    print('RMSE: %.3f' % rmse)
    print('MAPE: %.3f percent' % mape)
    print('Explained Variance: %.3f' % expl_var)
    print('R2: %.3f' % r2)

    # foobar = pd.DataFrame(plot_model.predict(X_test))
    # foobar.index = X_test.index

    foobar = pd.DataFrame(plot_model.predict(X))
    foobar.index = X.index

    # ax=sns.lineplot(data=foobar.resample('W').mean(), x=foobar.resample('W').mean().index, y=0, label='predicted')
    # ax=sns.lineplot(data=y.resample('W').mean(), label='actual', ax=ax)
    # ax.set_title(name)
    # ax.set_xlabel('Year')
    # ax.set_ylabel('Electricity Consumption (kWh)')
    # plt.show()

    axs[int(i/3),i%3]=sns.lineplot(ax=axs[int(i/3),i%3], data=foobar.resample('W').sum(), x=foobar.resample('W').sum().index, y=0, label='predicted')
    axs[int(i/3),i%3]=sns.lineplot(data=y.resample('W').sum(), label='actual', ax=axs[int(i/3),i%3])
    axs[int(i/3),i%3].set_title(name)
    axs[int(i/3),i%3].set_xlabel('')
    axs[int(i/3),i%3].set_ylabel('')
    axs[int(i/3),i%3].set(ylim=(-1000,45000))
    axs[int(i/3),i%3].set_xticks(pd.date_range('2013-01','2016-01-31', freq='AS'))

    i=i+1
axs[0,0].set_ylabel('Electricity Consumption (kWh)')
axs[1,0].set_ylabel('Electricity Consumption (kWh)')
axs[1,0].set_xlabel('Year')
axs[1,1].set_xlabel('Year')
axs[1,2].set_xlabel('Year')
axs[-1,-1].axis('off')
# ax.set_xticklabels(['Jan','Feb',"Mar",'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.tight_layout()
plt.show()
# plt.show()


#%% markdown
#%% codecell

fig, axs = plt.subplots(1, 2, figsize=(20,5))

X, y = new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 1 ][hourly_features].dropna().iloc[:,:-1], new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 1 ][hourly_features].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

plt.figure(figsize=(15,8))
axs[0]=sns.barplot(ax=axs[0], x=X.columns, y=plot_model.coef_)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=30, ha="right")
axs[0].set(ylim=(-60,50))
axs[0].set_title('Weekend data')


X, y = new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 0 ][hourly_features].dropna().iloc[:,:-1], new_combined_data_with_nans['2013-02':'2016-01-31'][ new_combined_data_with_nans['WEEKEND'] == 0 ][hourly_features].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

plt.figure(figsize=(15,8))
axs[1]=sns.barplot(ax=axs[1], x=X.columns, y=plot_model.coef_)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=30, ha="right")
axs[1].set_title('Weekday data')
axs[1].set(ylim=(-60,50))
plt.tight_layout()
plt.show()

# %%markdown
# %%codecell

fig, axs = plt.subplots(1, 3, figsize=(20,5))

X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

plt.figure(figsize=(15,8))
axs[0]=sns.barplot(ax=axs[0], x=X.columns, y=plot_model.coef_)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=33, ha="right")
axs[0].set_yscale('symlog')
axs[0].set(ylim=(-10e4,10e4))
axs[0].set_title('Hourly data')

X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

plt.figure(figsize=(15,8))
axs[1]=sns.barplot(ax=axs[1], x=X.columns, y=plot_model.coef_)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=33, ha="right")
axs[1].set_yscale('symlog')
axs[1].set(ylim=(-10e4,10e4))
axs[1].set_title('Daily data')

X, y = new_combined_data_with_nans_weekly[weekly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_weekly[weekly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

plt.figure(figsize=(15,8))
axs[2]=sns.barplot(ax=axs[2], x=X.columns, y=plot_model.coef_)
axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=33, ha="right")
axs[2].set_yscale('symlog')
axs[2].set(ylim=(-10e4,10e4))
axs[2].set_title('Weekly data')
new_combined_data_with_nans.dropna()['NO_WIND'].value_counts()
plt.tight_layout()
plt.show()



    X_train.columns
    X_train.columns[plot_model.coef_ != 0]
    plot_model.coef_[plot_model.coef_.argsort()]



    X_train.columns[plot_model.coef_.argsort()]
    weekday_features

# %%markdown
# %%codecell


fig, axs = plt.subplots(1, 3, figsize=(20,5))

X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

foobar = pd.DataFrame(plot_model.predict(X))
foobar.index = X.index

# plt.figure(figsize=(15,8))
axs[0]=sns.lineplot(ax=axs[0], data=foobar['2014-2'], x=foobar['2014-2'].index, y=0, label='predicted')
axs[0]=sns.lineplot(data=y['2014-2'], label='actual', ax=axs[0])
axs[0].set_title('Hourly data')
axs[0].set_xlabel('')
axs[0].set_ylabel('')
axs[0].set(ylim=(0,325))
axs[0].set_xticks(pd.date_range('2014-02','2014-3', freq='MS'))



X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

foobar = pd.DataFrame(plot_model.predict(X))
foobar.index = X.index

# plt.figure(figsize=(15,8))
axs[1]=sns.lineplot(ax=axs[1], data=foobar['2014-2'], x=foobar['2014-2'].index, y=0, label='predicted')
axs[1]=sns.lineplot(data=y['2014-2'], label='actual', ax=axs[1])
axs[1].set_title('Daily data')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].set(ylim=(0,5600))
axs[1].set_xticks(pd.date_range('2014-02','2014-3', freq='MS'))

X, y = new_combined_data_with_nans_weekly[weekly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_weekly[weekly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

foobar = pd.DataFrame(plot_model.predict(X))
foobar.index = X.index

# plt.figure(figsize=(15,8))
axs[2]=sns.lineplot(ax=axs[2], data=foobar['2014-2'], x=foobar['2014-2'].index, y=0, label='predicted')
axs[2]=sns.lineplot(data=y['2014-2'], label='actual', ax=axs[2])
axs[2].set_title('Weekly data')
axs[2].set_xlabel('')
axs[2].set_ylabel('')
axs[2].set(ylim=(0,37000))
axs[2].set_xticks(pd.date_range('2014-02','2014-3', freq='MS'))

plt.tight_layout()
plt.show()

# %%markdown
# %%codecell


fig, axs = plt.subplots(1, 2, figsize=(20,5))


X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().join( data_ups_combine_converted_filtered['Other'] )['Other']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

foobar = pd.DataFrame(plot_model.predict(X))
foobar.index = X.index

# plt.figure(figsize=(15,8))
axs[0]=sns.lineplot(ax=axs[0], data=foobar['2014-2'], x=foobar['2014-2'].index, y=0, label='predicted')
axs[0]=sns.lineplot(data=y['2014-2'], label='actual', ax=axs[0])
axs[0].set_title('Hourly data with UPS and Forum consumption removed')
axs[0].set_xlabel('')
axs[0].set_ylabel('')
axs[0].set(ylim=(0,325))
axs[0].set_xticks(pd.date_range('2014-02','2014-3', freq='MS'))


X, y = new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans[hourly_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().join( data_ups_combine_converted_filtered['Other'].resample('D').sum() )['Other']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

foobar = pd.DataFrame(plot_model.predict(X))
foobar.index = X.index

# plt.figure(figsize=(15,8))
axs[1]=sns.lineplot(ax=axs[1], data=foobar['2014-2'], x=foobar['2014-2'].index, y=0, label='predicted')
axs[1]=sns.lineplot(data=y['2014-2'], label='actual', ax=axs[1])
axs[1].set_title('Hourly data')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].set(ylim=(0,325))
axs[1].set_xticks(pd.date_range('2014-02','2014-3', freq='MS'))

plt.tight_layout()
plt.show()

#%% markdown
#%% codecell

fig, axs = plt.subplots(1, 2, figsize=(20,5))

X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().join( data_ups_combine_converted_filtered['Other'].resample('D').sum() )['Other']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)

plt.figure(figsize=(15,8))
axs[0]=sns.barplot(ax=axs[0], x=X.columns, y=plot_model.coef_)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=30, ha="right")
axs[0].set(ylim=(-2000,1500))
axs[0].set_title('Hourly data with UPS and Forum consumption removed')


X, y = new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna().drop('bin_kwh', axis=1), new_combined_data_with_nans_daily[daily_features]['2013-02':'2016-01-31'].dropna()['bin_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

plot_model = LinearRegression();
plot_model.fit(X, y)
X.columns
plot_model.coef_
plt.figure(figsize=(15,8))
axs[1]=sns.barplot(ax=axs[1], x=X.columns, y=plot_model.coef_)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=30, ha="right")
axs[1].set_title('Hourly data')
axs[1].set(ylim=(-2000,1500))
plt.tight_layout()
plt.show()

# %%markdown
# %%codecell

# from tsfresh import extract_relevant_features
#
# features_filtered_direct = extract_relevant_features(X.reset_index(), y,
#                                                      column_id='OB_TIME')
#
#
# features_filtered_direct
#
#
# my_model = LinearRegression()
# my_model.fit(features_filtered_direct, y)
#
#
# # evaluate the model
# yhat = my_model.predict(features_filtered_direct)
# # evaluate predictions
# rmse = mean_squared_error(y, yhat)**0.5
# print('RMSE: %.3f' % rmse)
#
# ax=sns.lineplot(x=features_filtered_direct.index, y=my_model.predict(features_filtered_direct))
#
# y.plot(ax=ax)
#
# #%% markdown
# #%% codecell
#
# features_filtered_direct_2 = extract_relevant_features(new_combined_data_with_nans[ [col for col in new_combined_data_with_nans.columns if col != 'bin_kwh' ] ].dropna().reset_index(), new_combined_data_with_nans.dropna()['bin_kwh'],
#                                                      column_id='OB_TIME')
#
#
# features_filtered_direct_2
#
#
# my_model = LinearRegression()
# my_model.fit(features_filtered_direct, y)
#
#
# # evaluate the model
# yhat = my_model.predict(features_filtered_direct)
# # evaluate predictions
# rmse = mean_squared_error(y, yhat)**0.5
# print('RMSE: %.3f' % rmse)
#
# ax=sns.lineplot(x=features_filtered_direct.index, y=my_model.predict(features_filtered_direct))
#
# y.plot(ax=ax)
#
#
# # minus UPS;
# # monthly;
