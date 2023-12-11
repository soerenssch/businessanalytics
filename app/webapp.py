#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 13:00:12 2023

@author: sören
"""

### packages

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import statsmodels
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


### functions

def generate_predictions(model_name):
    if model_name == 'XGBoost':
        XGBoost = pickle.load(open('models/models_sav/XGB_grid.sav', 'rb'))
        predictions = XGBoost.predict(X)

        pass
    elif model_name == 'Option2':
        # Generate predictions using model2
        pass
    # Add more models as needed
    return predictions


## past data and trained models

df_train = pd.read_csv('data/custom/df_train.csv')

# Preprocess data for predictions
df_train['DATETIME'] = pd.to_datetime(df_train['DATETIME'])

df_daily = df_train[['DATETIME', 'ND_TARGET']]

df_daily.set_index('DATETIME', inplace=True)
df_daily = df_daily[['ND_TARGET']].resample('D').sum()

X = df_train.drop(['ND_TARGET'], axis=1)
y = df_train['ND_TARGET']

X['year'] = X['DATETIME'].dt.year
X['month'] = X['DATETIME'].dt.month
X['day'] = X['DATETIME'].dt.day
X['hour'] = X['DATETIME'].dt.hour
X = X.drop(['DATETIME'], axis=1)


model_options = ['XGBoost', 'SARIMAX', 'Option3']


### Streamlit configurations

st.title('Forecasting UK Electricity demand')

tab1, tab2, tab3 = st.tabs(["Scenario modelling", "Live data", 'Past predictions/Model evaluation'])

with tab1: # Scenario inputs
    st.write('Scenario inputs')

    row1_col1, row1_col2, row1_col3 = st.columns([2.5,2.5,2.5]) 
    Day = row1_col1.slider('Select Forecast length (in days).', 1, 100, 30, key=9)
    Backtesting = row1_col2.slider('How many days backwards?', 1, 500, 100, key=10)
    Weather = row1_col3.slider('Select the weather.', 0.0, 2.0, 0.6, key=11)


    stl = STL(df_daily, seasonal=365)
    res = stl.fit()
    seasonal = res.seasonal
    trend = res.trend
    residual = res.resid

    residual_model = SARIMAX(residual, order=(5, 0, 1), seasonal_order=(0, 0, 0, 0)) 
    residual_model = residual_model.fit()

    # Exponential Smoothing for trend prediction
    trend_model = ExponentialSmoothing(trend, trend='add', seasonal=None, damped_trend=True)
    trend_model = trend_model.fit(smoothing_level=0.6, smoothing_trend=0.02, damping_trend=1)

    # Forecast the trend component out-of-sample
    trend_forecast = trend_model.predict(start=len(df_daily), end=len(df_daily) + Day - 1)

    # Extend the seasonal component by repeating the last year's seasonality
    seasonal_forecast = np.tile(seasonal[-Day:], 1)
    seasonal_forecast = seasonal_forecast[:Day]

    # Forecast the residuals out-of-sample
    residual_forecast = residual_model.get_forecast(steps=Day).predicted_mean

    # Combine the forecast components
    combined_forecast = trend_forecast.values + seasonal_forecast + residual_forecast

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df_daily.tail(Backtesting), label='Past')
    ax.plot(combined_forecast, label='Forecast')
    ax.set_title('Past vs Forecasted Energy Demand')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Demand')
    ax.legend()
    st.pyplot(fig)


with tab2: # Live data via API
    st.write('API Calls for live data')


with tab3: # Model evaluation
    selected_models = st.multiselect('Select models to evaluate', model_options)

    for model_name in selected_models:

        if model_name == 'SARIMAX':

            stl = STL(df_daily, seasonal=365)
            res = stl.fit()
            seasonal = res.seasonal
            trend = res.trend
            residual = res.resid

            residual_model = SARIMAX(residual, order=(5, 0, 1), seasonal_order=(0, 0, 0, 0)) 
            residual_model = residual_model.fit()
            predictions = residual_model.predict(start=0, end=len(residual)-1) + trend + seasonal

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(df_daily, label='Observed')
            ax.plot(predictions, label='Predictions')
            ax.set_title('In-Sample prediction for SARIMAX')
            ax.set_xlabel('Date')
            ax.set_ylabel('Energy Demand')
            ax.legend()
            st.pyplot(fig)
        
        else:
            predictions = generate_predictions(model_name)

            results = pd.DataFrame({
                'DATETIME': df_train['DATETIME'],
                'Actual': y,
                'Predicted': predictions
            })

            # plt.figure(figsize=(10, 10))
            # sns.scatterplot(x='Actual', y='Predicted', data=results)
            # plt.title(f'Predicted vs Actual for {model_name}')
            # plt.xlabel('Actual Values')
            # plt.ylabel('Predicted Values')
            # plt.plot([results['Actual'].min(), results['Actual'].max()], 
            #          [results['Actual'].min(), results['Actual'].max()], 
            #          color='red', linestyle='--')
            # st.pyplot(plt)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # First plot
            ax1.plot(results['DATETIME'], results['Actual'], label='Actual', color='blue')
            ax1.plot(results['DATETIME'], results['Predicted'], label='Predicted', color='orange', alpha=0.5)
            ax1.set_title('Actual vs Predicted Energy Demand')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Energy Demand')
            ax1.legend()

            # Second plot
            sns.scatterplot(x='Actual', y='Predicted', data=results, ax=ax2)
            ax2.set_title(f'Predicted vs Actual for {model_name}')
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.plot([results['Actual'].min(), results['Actual'].max()], 
                    [results['Actual'].min(), results['Actual'].max()], 
                    color='red', linestyle='--')
            
            st.pyplot(fig)
