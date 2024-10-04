import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle as pkl
import time
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def retrieve_data(uri, db_name, collection_names, save=False):
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[db_name]
    all_docs = []

    for collection_name in collection_names:
        collection_name = "_".join(collection_name.lower().split())
        collection = db[collection_name]
        docs = collection.find({})
        model_docs = []
        for doc in docs:
            doc.pop('_id', None)
            model_docs.append(doc)
            all_docs.append(doc)
        model_data = pd.DataFrame(model_docs)
        if save:
            model_data.to_csv(f'all_{collection_name}_data.csv', index=False)

    all_data = pd.DataFrame(all_docs)
    if save:
        all_data.to_csv(f'all_{db_name}_data.csv', index=False)

    client.close()
    return all_data

def preprocess_data(df, recent_seconds=None, recent_minutes=None, recent_hours=None, recent_days=None):
    if 'timestamp' not in df.columns:
        st.error("The data does not contain a 'timestamp' column.")
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    if recent_seconds is not None:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(seconds=recent_seconds)
        cutoff = cutoff.to_datetime64()
        df = df[df['timestamp'] >= cutoff]
    elif recent_minutes is not None:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=recent_minutes)
        cutoff = cutoff.to_datetime64()
        df = df[df['timestamp'] >= cutoff]
    elif recent_hours is not None:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=recent_hours)
        cutoff = cutoff.to_datetime64()
        df = df[df['timestamp'] >= cutoff]
    elif recent_days is not None:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
        cutoff = cutoff.to_datetime64()
        df = df[df['timestamp'] >= cutoff]
    
    return df

@st.cache_resource
def load_model(model):
    return pkl.load(open(f'ml_models/{"".join(model.lower().split())}_model.pkl', 'rb'))

st.set_page_config(
    page_title='Vehicle Data Monitor',
    page_icon='📈',
    layout='wide'
)

st.image('mtec.logo.png', caption='Engineering moves you to the future', width=220)
st.write('\n')

uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"

column_1, column_2 = st.columns(2)
with column_1:
    st.write('## Vehicle Data Monitor 📈')
    mean_placeholder = st.empty()
with column_2:
    c1, c2, c3 = st.columns(3)
    with c1:
        time_period = st.selectbox("Select Time Period", ['Seconds', 'Minutes', 'Hours', 'Days'])
    with c2:
        models_df = retrieve_data(uri, 'vehicles', ['models'])
        model_options = models_df['model'].dropna().unique()
        a = {k.replace('Model', 'Vehicle'): k for k in model_options}
        model = st.selectbox("Select Model", list(a.keys()))
        model = a[model]
    with c3:
        ml_models = ['Linear Regression', 'SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting']
        selected_ml_model = st.selectbox('Select ML Model', ml_models)
    ml_model = load_model(selected_ml_model)
    individual_placeholder = st.empty()

while True:
    whole_df = retrieve_data(uri, 'vehicles', model_options)
    with mean_placeholder.container():
        current_col, voltage_col = st.columns(2)
        with current_col:
            mean_current_health = whole_df.groupby('model')['current'].mean().reset_index()
            mean_current_health['model'] = mean_current_health['model'].apply(lambda x: x.replace('Model', 'Vehicle'))
            fig = px.bar(mean_current_health, x='model', y='current', text='current', color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Vehicles',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Current',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                )
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
            st.write(fig)
        with voltage_col:
            mean_voltage = whole_df.groupby('model')['voltage'].mean().reset_index()
            mean_voltage['model'] = mean_voltage['model'].apply(lambda x: x.replace('Model', 'Vehicle'))
            fig = px.bar(mean_voltage, x='model', y='voltage', text='voltage', color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Vehicles',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Voltage',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                )
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
            st.write(fig)
        temperature_col, battery_health_col = st.columns(2)
        with temperature_col:
            mean_temperature = whole_df.groupby('model')['temperature'].mean().reset_index()
            mean_temperature['model'] = mean_temperature['model'].apply(lambda x: x.replace('Model', 'Vehicle'))
            fig = px.bar(mean_temperature, x='model', y='temperature', text='temperature', color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Vehicles',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Temperature',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                )
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
            st.write(fig)
        with battery_health_col:
            mean_battery_health = whole_df.groupby('model')['battery_health'].mean().reset_index()
            mean_battery_health['model'] = mean_battery_health['model'].apply(lambda x: x.replace('Model', 'Vehicle'))
            fig = px.bar(mean_battery_health, x='model', y='battery_health', text='battery_health', color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Vehicles',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Battery Health',
                    title_font={'color': 'black'},
                    tickfont_color='black'
                )
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
            st.write(fig)
    df = retrieve_data(uri, 'vehicles', [model])
    if time_period == 'Seconds':
        df_preprocessed = preprocess_data(df, recent_seconds=10)
    elif time_period == 'Minutes':
        df_preprocessed = preprocess_data(df, recent_minutes=1)
    elif time_period == 'Hours':
        df_preprocessed = preprocess_data(df, recent_hours=1)
    elif time_period == 'Days':
        df_preprocessed = preprocess_data(df, recent_days=1)
    else:
        df_preprocessed = preprocess_data(df)
    if df_preprocessed.empty:
        break
    df_preprocessed['battery_health_pred'] = ml_model.predict(df_preprocessed[['voltage', 'current', 'temperature']])
    with individual_placeholder.container():
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            fig = px.line(df_preprocessed, x='timestamp', y='voltage', labels={'timestamp': 'Timestamp', 'voltage': 'Voltage (V)'}, color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=200,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Time',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Voltage',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                )
            )
            st.write(fig)
            fig = px.line(df_preprocessed, x='timestamp', y='current', labels={'timestamp': 'Timestamp', 'current': 'Current (A)'}, color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=200,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Time',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Current',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                )
            )
            st.write(fig)
        with fig_col2:
            fig = px.line(df_preprocessed, x='timestamp', y='temperature', labels={'timestamp': 'Timestamp', 'temperature': 'Temperature (˚C)'}, color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=200,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Time',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Temperature',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                )
            )
            st.write(fig)
            fig = px.line(df_preprocessed, x='timestamp', y='battery_health', labels={'timestamp': 'Timestamp', 'battery_health': 'Battery Health (%)'}, color_discrete_sequence=['#3EBABC'])
            fig.update_layout(
                height=200,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title='Time',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                ),
                yaxis=dict(
                    title='Battery Health',
                    title_font={'color': 'black'},
                    showgrid=True,
                    gridcolor='gray',
                    tickfont_color='black'
                )
            )
            st.write(fig)
        fig = px.line(df_preprocessed, x='timestamp', y='battery_health_pred', labels={'timestamp': 'Timestamp', 'battery_health': 'Battery Health (%)'}, color_discrete_sequence=['#3EBABC'])
        fig.update_layout(
            height=200,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title='Time',
                title_font={'color': 'black'},
                showgrid=True,
                gridcolor='gray',
                tickfont_color='black'
            ),
            yaxis=dict(
                title='Battery Health\nEstimation',
                title_font={'color': 'black'},
                showgrid=True,
                gridcolor='gray',
                tickfont_color='black'
            )
        )
        st.write(fig)
    time.sleep(1)