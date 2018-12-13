#!/usr/bin/env python
# coding: utf-8

# load the packages
# please note that this code is not an exact replication, but with 3 things to keep in mind: 
# 1. betas ex ante are estimated based on monthly(not daily) data for computational reasons, so it can be easily run in the class
# 2. no smoothing of returns is applied (correction for non-syncronous trading), which stems from point #1
# 3. It calculates alphas wrt CAPM (no fama-french)
import pandas as pd
import os
import numpy as np
import datetime
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm



# set up a working directory, this is where all the data should be
os.chdir('/Users/PUT_HERE_THE_PATH')
# display options for pandas, we don't want to see 1000 rows
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


#load mkt returns (those are already over a risk-free rate)
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


#load rf returns
rf = pd.read_csv('rf_monthly.csv')
rf.head()
# Again recognizing the date format
rf['dateff'] = pd.to_datetime(rf['dateff'], format='%Y%m%d')
# Renaming the columns
rf.columns = ['date', 'rf']
#Converting dates into monthly period
rf['date'] = rf['date'].dt.to_period('M')


# load the data into dataframe
# this code does replication on monthly returns (instead of daily as in the paper, but can be easily changed)
df = pd.read_csv('CRSP_monthly_1926_2014.csv')
df.head()

df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.columns = ['permno', 'date', 'shrcd', 'ret']
# Here inside the square brackets we try to convert 'ret' to numeric values (sometimes u can see letters 
# or non-numeric signs)
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
print(df)
# rename columns
df.columns = ['date', 'id', 'ret']


# To calculate betas we need to merge the main dataset with two other datasets: rf rate and mkt returns
# mkt
df = pd.merge(df, mkt, on='date', how='left')
# rf
df = pd.merge(df, rf, on='date', how='left')
# HERE is important, we run regressions on EXCESS returns, not gross returns
# in mkt dataset returns are ALREADY over risk-free rate
df['ret'] = df['ret'] - df['rf']


# define function to estimate rolling 5 year(60 month) correlations with minimum 36 non-missing datapoints
def roll_corr(x):
    return pd.DataFrame(x['ret'].rolling(60, min_periods=36).corr(x['mkt']))


#same for rolling std, but with 1 year horizon
def roll_var(x):
    return pd.DataFrame(x['ret'].rolling(12, min_periods=12).std())


#Then we need to apply this functions to each stock, for this purpose we use groupby 'id'
# Shift is used so that the current datapoint is not included
df['corr_est'] = df.groupby('id')[['ret', 'mkt']].apply(roll_corr)
df['corr_est'] = df.groupby('id')['corr_est'].shift(1)
df['id_var_est'] = df.groupby('id')[['ret', 'mkt']].apply(roll_var)
df['id_var_est'] = df.groupby('id')[['id_var_est']].shift(1)


#drop all the rows where in ANY column there is a NAN value
df = df.dropna(how='any')
# Estimation betas like on page 8 in eq (14) in the paper
df['beta_est'] = df['corr_est']*df['id_var_est'].div(df['std_est'])
#Shrink the betas to make them less noisy eq(15)
df['beta_est'] = 0.6*df['beta_est'] + 0.4
# Then we assign stocks into portfolios based on the their beta_est quantile
df['q'] = df.groupby('date')['beta_est'].apply(lambda x: pd.qcut(x, 10, labels=range(1, 11)))


# check the average mean excess return and average estimated beta and compare it with the table 3
print(df.groupby('q')[['beta_est', 'ret']].mean())
alpha = df.groupby(['date', 'q'])[['ret', 'mkt']].mean().reset_index()
alpha.columns = ['date', 'q', 'ret', 'mkt']


#Create an empty dataframe to store estimated alphas and betas
par = pd.DataFrame()
for i in range(1,11):
    # pick the portfolio
    alpha0 = alpha[alpha['q']==i]
    x = alpha0['mkt'].copy()
    # add a constant to the model, default ols goes without it
    x = sm.add_constant(x)
    #estimate the model
    results = smf.OLS(alpha0['ret'], x).fit(cov_type='HC1')
    #print the table with estimates
    print(results.summary())
    # get alphas and betas
    par0 = results.params
    # give a name to the row of parameters, we need to know for which portfolio we got the estimates
    par0.name= 'port_{}'.format(i)
    # join it with the dataframe of params
    par = par.append(par0)
par.columns = ['alpha', 'beta_realised']
print(par)

