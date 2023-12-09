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



### functions

def generate_predictions(model_name):
    if model_name == 'XGBoost':
        XGBoost = pickle.load(open('../models/models_sav/XGB_grid.sav', 'rb'))
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

X = df_train.drop(['ND_TARGET'], axis=1)
y = df_train['ND_TARGET']

X['year'] = X['DATETIME'].dt.year
X['month'] = X['DATETIME'].dt.month
X['day'] = X['DATETIME'].dt.day
X['hour'] = X['DATETIME'].dt.hour
X = X.drop(['DATETIME'], axis=1)


model_options = ['XGBoost', 'Option2', 'Option3']


### Streamlit configurations

st.title('Forecasting UK Electricity demand')

tab1, tab2, tab3 = st.tabs(["Scenario modelling", "Live data", 'Past predictions/Model evaluation'])

with tab1: # Scenario inputs
    st.write('Scenario inputs')

    row1_col1, row1_col2, row1_col3 = st.columns([2.5,2.5,2.5]) 
    Day = row1_col1.slider('Select a day.', 0.0, 0.5, 0.007, key=9)
    Month = row1_col2.slider('Select a month.', 0.0, 1.0, 0.05, key=10)
    Weather = row1_col3.slider('Select the weather.', 0.0, 2.0, 0.6, key=11)


with tab2: # Live data via API
    st.write('API Calls for live data')


with tab3: # Model evaluation
    selected_models = st.multiselect('Select models to evaluate', model_options)

    for model_name in selected_models:
        predictions = generate_predictions(model_name)

        results = pd.DataFrame({
            'DATETIME': df_train['DATETIME'],
            'Actual': y,
            'Predicted': predictions
        })

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='Actual', y='Predicted', data=results)
        plt.title(f'Predicted vs Actual for {model_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([results['Actual'].min(), results['Actual'].max()], 
                 [results['Actual'].min(), results['Actual'].max()], 
                 color='red', linestyle='--')
        st.pyplot(plt)
