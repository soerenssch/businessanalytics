import streamlit as st
from ElexonDataPortal import api
from secret import api_key
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta, date
import time
import schedule
import pickle
import numpy as np

client = api.Client(api_key)

@st.cache_data
def get_time():
    return datetime.utcnow()

@st.cache_data
def fetch_demand_data(start_date, end_date):
    df = client.get_SYSDEM(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return df

@st.cache_data
def retrain_model_and_forecast():
    model = pickle.load(open('./model/tbats_model.pkl', 'rb'))
    df = client.get_SYSDEM(start_date=model.trained_data.index[-1], end_date=date.today())
    df.index = pd.to_datetime(df['local_datetime'])
    df = df.loc[df['recordType'] == 'ITSDO']
    s = df['demand'].astype('float64')
    new_index = pd.date_range(start=(model.trained_data.index[-1] + timedelta(minutes=30)), end=s.index[-1].astimezone('UTC'), freq='30T', tz='UTC')
    s_wo_duplicate = s.loc[~s.index.duplicated(keep='first')]
    s_reindex = s_wo_duplicate.reindex(new_index)
    s_reindex_interp = s_reindex.interpolate(method='linear')
    s_new_data = pd.concat([model.trained_data, s_reindex_interp])
    model.fit(s_new_data)
    model.trained_data = s_new_data
    pickle.dump(model, open('./model/tbats_model.pkl', 'wb'))
    forecast_dp = 48*30
    forecast_index = pd.date_range(start=model.trained_data.index[-1], periods=forecast_dp+1, freq='30T', tz='UTC')
    forecast = np.insert(model.forecast(forecast_dp), 0, model.trained_data.iloc[-1])
    return pd.Series(forecast, index=forecast_index)



def refresh():

    time = get_time()
    df = fetch_demand_data((time - timedelta(days=history_days)), (time + timedelta(days=forecast_days)))
    forecast_series = retrain_model_and_forecast()

    forecast_dataframe = forecast_series.to_frame(name='demand').reset_index(names='local_datetime')
    forecast_dataframe['recordType'] = 'our_forecast'

    new_df = pd.concat([df, forecast_dataframe])

    with market_price_placeholder.container():

        chart = px.line(new_df, x="local_datetime", y="demand", color='recordType', labels={
                     "local_datetime": "Time (UTC)",
                     "gen": "Generated Electricity in MW",
                     "fuel": "Fuel Type"
                 }, title='Generation by Fuel Type')
        chart.update_xaxes(range=[(time - timedelta(days=history_days)).strftime("%Y-%m-%d"), (time + timedelta(days=(forecast_days+1))).strftime("%Y-%m-%d")])
        st.plotly_chart(chart, use_container_width=True)

        st.write(f'Last Update: {get_time().strftime("%H:%M")}')


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