#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:45:00 2017

@author: Husna
"""

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from datetime import datetime

def execute():
    " This code will run entire script in a step by step method "
    
    # First we will read the data into python
    sessions = read_sessions()
    engagements = read_engagements()
    print(sessions.head())
    print(engagements.head())
    
    # Next we have to transform the datasets are both are in different formats
    engagements = get_engagements(engagements)
    df = merge_to_user_id(sessions, engagements)   # merge on user_id
    df = delete_remaining(df)
    df = add_conversion_metric(df)
    df = add_pageviews_cum_sum(df)
    df.to_csv('output\\df_transformed.csv')     # outputs the new merged csv

    # Using logistic regression, we will fit the model 
    logistic_regression_results = run_logistic_regression(df)
    predict_probabilities(logistic_regression_results)
    
    # Visualizing results
    visualize_results(df)
                          
 
def read_sessions():
    sessions = pd.read_csv('sessions.csv', parse_dates={'datetime': [2]})    # fixed the session_start_date column
    
    # we need user_id to join both dataframes, need datetime & session number to search for first engagement and sessions before that 
    # need pageviews to test hypothesis                                             
    columns_needed = ['datetime', 'user_id', 'session_number', 'pageviews']              
    
    sessions = sessions[columns_needed]
    return sessions
    
def read_engagements():
    engagements = pd.read_csv('engagements2.csv', 
                              parse_dates={'datetime': [2]}, 
                              date_parser=parse_unixtstamp_as_datetime)
    
    # we only need 'datetime' and 'user_id' here
    columns_needed = ['datetime', 'user_id']
    engagements = engagements[columns_needed]
    return engagements
    
def parse_unixtstamp_as_datetime(unix_tstamp):
    datetime_obj = datetime.fromtimestamp(int(unix_tstamp)) # converting to an integer
    return datetime_obj
    
def get_engagements(engagements):
    # will filter to only get teh first engagement of each user
    engagements.sort_values(['user_id', 'datetime'], ascending=True, inplace=True)
    engagements = engagements.groupby(['user_id'])['datetime'].first().reset_index()
    return engagements
    
def merge_to_user_id(sessions, engagements):
    # this will merge the data sets on user_id
    df = pd.merge(sessions,
                  engagements,
                  on='user_id',
                  how='left',
                  suffixes=('_session', '_first_engagement'))
    return df
    
def delete_remaining(df):
    # deletes sessions after the first engagement, not needed for this analysis 
    condition = df['datetime_first_engagement'] >= df['datetime_session']
    df = df[condition].copy()
    return df
    
def add_conversion_metric(df):
    # this will add conversion metric
    df['is_conversion'] = False
    # get row indices of sessions with engagements and set is_conversion to true
    indices = df.groupby(['user_id']).apply(lambda x: x['datetime_session'].idxmax())
    df.ix[indices, 'is_conversion'] = True
    return df
    
def add_pageviews_cum_sum(df):
    # Add cumulative sum of pageviews
    df['pageviews_cum_sum'] = df.groupby('user_id')['pageviews'].cumsum()
    return df
    
def run_logistic_regression(df):
    # Logistic regression
    X = df['pageviews_cum_sum']
    X = sm.add_constant(X)
    y = df['is_conversion']
    logit = sm.Logit(y, X)
    logistic_regression_results = logit.fit()
    print(logistic_regression_results.summary())
    return logistic_regression_results
    
def predict_probabilities(logistic_regression_results):
    # Predict the conversion probability for 0 up till 50 pageviews
    X = sm.add_constant(range(0, 50))
    y_hat = logistic_regression_results.predict(X)
    df_hat = pd.DataFrame(zip(X[:, 1], y_hat))
    df_hat.columns = ['X', 'y_hat']
    p_conversion_25_pageviews = df_hat.ix[25]['y_hat']
    print("")
    print("The probability of converting after 25 pageviews is {}".format(p_conversion_25_pageviews))


def visualize_results(df):
    # Visualize logistic curve using seaborn
    sns.set(style="darkgrid")
    sns.regplot(x="pageviews_cumsum",
                y="is_conversion",
                data=df,
                logistic=True,
                n_boot=500,
                y_jitter=.01,
                scatter_kws={"s": 60})
    sns.set(font_scale=1.3)
    sns.plt.title('Logistic Regression Curve')
    sns.plt.ylabel('Conversion probability')
    sns.plt.xlabel('Cumulative sum of pageviews')
    sns.plt.subplots_adjust(right=0.93, top=0.90, left=0.10, bottom=0.10)
    sns.plt.show()
    
# Run the whole code
execute()    

















