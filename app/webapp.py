#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 13:00:12 2023

@author: sören
"""

### packages

import streamlit as st
import pandas as pd


### functions



### Streamlit configurations

st.title('Forecasting UK Electricity demand')

tab1, tab2 = st.tabs(["Scenario modelling", "Live data"])

with tab1: # Scenario inputs
    st.write('Scenario inputs')

    row1_col1, row1_col2, row1_col3 = st.columns([2.5,2.5,2.5]) 
    Day = row1_col1.slider('Select a day.', 0.0, 0.5, 0.007, key=9)
    Month = row1_col2.slider('Select a month.', 0.0, 1.0, 0.05, key=10)
    Weather = row1_col3.slider('Select the weather.', 0.0, 2.0, 0.6, key=11)


with tab2: # Live data via API
    st.write('API Calls for live data')