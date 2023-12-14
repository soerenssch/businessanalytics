import streamlit as st
from ElexonDataPortal import api
from secret import api_key
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import schedule

client = api.Client(api_key)

@st.cache_data
def get_time():
    return datetime.utcnow()

@st.cache_data
def fetch_demand_data(start_date, end_date):
    df = client.get_SYSDEM(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return df

def refresh():

    time = get_time()
    df = fetch_demand_data((time - timedelta(days=history_days)), (time + timedelta(days=forecast_days)))

    with market_price_placeholder.container():

        chart = px.line(df, x="local_datetime", y="demand", color='recordType', labels={
                     "local_datetime": "Time (UTC)",
                     "gen": "Generated Electricity in MW",
                     "fuel": "Fuel Type"
                 }, title='Generation by Fuel Type')
        chart.update_xaxes(range=[(time - timedelta(days=history_days)).strftime("%Y-%m-%d"), (time + timedelta(days=(forecast_days+1))).strftime("%Y-%m-%d")])
        st.plotly_chart(chart, use_container_width=True)


st.set_page_config(page_title="UK Energy Dashboard", layout="wide", page_icon="📈")

st.title('Demand Forecast')

st.header('Settings')

col1, col2 = st.columns(2)
with col1: 
    history_days = st.select_slider(
        'Select Number of Past Days',
        options=[0, 7, 14, 28])
with col2:
    forecast_days = st.select_slider(
        'Select Number of Forecasted Days',
        options=[0, 7, 14, 28])



market_price_placeholder = st.empty()
refresh()



def refresh_data():
    st.cache_data.clear()
    refresh()

schedule.every().hour.at(':00').do(refresh_data)
schedule.every().hour.at(':30').do(refresh_data)

while 1:
    n = schedule.idle_seconds()
    if n is None:
        # no more jobs
        break
    elif n > 0:
        # sleep exactly the right amount of time
        time.sleep(n)
    schedule.run_pending()