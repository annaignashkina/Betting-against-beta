#!/usr/bin/env python
# coding: utf-8
# load the packages
import pandas as pd
import os
import numpy as np
import scipy
import datetime
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# set up a working directory
os.chdir('/Users/anna/Documents/Teaching/Betting against beta')
# display options for pandas, we don't want to see 1000 rows, though sometimes we do
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


# load monthly market returns
mkt = pd.read_excel('mkt_monthly.xlsx')
# make pandas recognize the dates
mkt['date'] = pd.to_datetime(mkt['date'], format='%m/%d/%Y')
# convert into monthly periods e.g 15-09-2018=>>>> 09-2018
mkt['date'] = mkt['date'].dt.to_period('M')


#calculate rolling standard deviation
# shift below is used to use PREVIOUS 12 month
mkt['std_est'] = mkt['mkt'].rolling(12).std().shift(1)
#Without shifting we get the REALIZED 12 month rolling std (though we will not use it)
mkt['std_real'] = mkt['mkt'].rolling(12).std()


#load rf monthly returns
rf = pd.read_csv('rf_monthly.csv')
rf.head()
# Again recognizing the date format
rf['dateff'] = pd.to_datetime(rf['dateff'], format='%Y%m%d')
# Renaming the columns
rf.columns = ['date', 'rf']
#Converting dates into monthly periods
rf['date'] = rf['date'].dt.to_period('M')


# load the returns data into dataframe
df = pd.read_csv('CRSP_monthly_1926_2014.csv')
# show first 5 lines of it
df.head()
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.columns = ['permno', 'date', 'shrcd', 'ret']
# Here inside the square brackets we try to convert 'ret' to numeric values (sometimes u can see letters or non-numeric signs)
df = df[pd.to_numeric(df['ret'], errors='coerce').notnull()]
df['ret'] = df['ret'].astype(float)
# Just in case covert stock ids into text format
df['permno'] = df['permno'].astype(str)
#create a column with the year (will not use it, just for general info)
df['year'] = df['date'].dt.year


df['date'] = df['date'].dt.to_period('M')
# In CRSP if a stock didn't trade at a particular date, there is no record about returns at that date at all.
# There is a requirement that each stock must have at least 12/36 month data point during previous 12/60 month
# to calculate betas. Thus, even if a stock didn't trade, we need to have a record like "permno, date, NAN, NAN.."
# to use rolling window and set a minimum requirement for the window.
df = df.pivot_table(index='date', values='ret', columns='permno')
# we want back our initial structure of the data
df = df.stack(dropna=False).reset_index()
# rename columns
df.columns = ['date', 'id', 'ret']


# To calculate betas we need to merge the main dataset with two other datasets: rf rate and mkt returns
# mkt
df = pd.merge(df, mkt, on='date', how='left')
#rf
df = pd.merge(df, rf, on='date', how='left')
# HERE is important, we run regressions on EXCESS returns, not gross returns
# in mkt dataset returns are ALREADY over risk-free rate
df['ret'] = df['ret'] - df['rf']


# define function to estimate rolling 5 year(60 month) correlations with minimum 36 non-missing datapoints
def roll_corr(x):
    return pd.DataFrame(x['ret'].rolling(60, min_periods=36).corr(x['mkt']))


#same for rolling std, but with 1 year horizon
def roll_std(x):
    return pd.DataFrame(x['ret'].rolling(12, min_periods=12).std())


#Then we need to apply this functions to each stock, for this purpose we use groupby 'id'
# Shift is used so that the current datapoint is not included
df['corr_est'] = df.groupby('id')[['ret', 'mkt']].apply(roll_corr)
df['corr_est'] = df.groupby('id')['corr_est'].shift(1)
df['id_std_est'] = df.groupby('id')[['ret', 'mkt']].apply(roll_std)
df['id_std_est'] = df.groupby('id')[['id_std_est']].shift(1)


#drop all the rows where in ANY column there is a NAN value
df = df.dropna(how='any')
# Estimation betas like on page 8 in eq (14) in the paper
df['beta_est'] = df['corr_est']*df['id_sdt_est'].div(df['std_est'])
#Shrink the betas to make them less noisy eq(15)
df['beta_est'] = 0.6*df['beta_est'] + 0.4


# For each month we devide betas into 2 groups: high(1) and low(0), cutoo is a median beta
df['q'] = df.groupby('date')['beta_est'].apply(lambda x: pd.qcut(x, 2, labels=range(0, 2)))


# for each month we rank betas and calculate the weights like in eq(16) in the paper
df['rank'] = df.groupby('date')['beta_est'].rank()
# z_bar
df['rank_avg'] = df.groupby('date')['rank'].transform('mean')
#abs(z-z_bar)
df['weight'] = abs(df['rank']-df['rank_avg'])


# calculate constant k
df['k'] = 2/(df.groupby('date')['weight'].transform('sum')).copy()
print(df)
# calculate final weights
df['weight'] = df['weight']*df['k']

# check the weights sum up to 1! Always doublecheck what you doublechecked and then check again
print(df.groupby(['date', 'q'])['weight'].sum())


# calculating beta_H and beta_L
df['dot'] = (df['beta_est']*df['weight'])
# calculating r_H and r_L
df['ret'] = (df['ret']*df['weight'])
# for each date and group (H, L) calculate the aggregate betas and rets
bab_beta = df.groupby(['date', 'q'])['dot', 'ret'].sum().reset_index()


bab_beta['inv'] = 1/bab_beta['dot']
print(bab_beta.groupby('q')['inv'].mean())


# weight multiplied by return
bab_beta['w_r'] = bab_beta['inv']*bab_beta['ret']


# just a trick to actualy get the difference by summation(r_bab=part_L-part_H)
bab_beta['w_r0'] = bab_beta.apply(lambda x: x['w_r'] if x['q']==0 else -x['w_r'], axis=1)
bab = bab_beta.groupby('date')['w_r0'].sum().reset_index()


# After groupby many columns disappear, need to merge with mkt dataset again
bab = pd.merge(bab, mkt, on='date', how='left')



# running regression to estimate alphas and betas
# x is mkt return
x = bab['mkt'].copy()
# this way you add a constant into the model (alpha basically), just a technicality
x = sm.add_constant(x)
results = smf.OLS(bab['w_r0'], x).fit(cov_type='HC1')
print(results.summary())




