import os
import streamlit as st
import pandas as pd
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
import pickle as pkl
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title='Mea-Tec',
    layout='wide',
)

if 'fetch_counter' not in st.session_state:
    st.session_state['fetch_counter'] = 0

@st.cache_resource
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
            model_data.to_csv(f'ml_models/all_{collection_name}_data.csv', index=False)

    all_data = pd.DataFrame(all_docs)
    if save:
        all_data.to_csv(f'all_{db_name}_data.csv', index=False)

    client.close()
    return all_data

def preprocess_data(data, impute_strategy='mean', num_records=10):
    # print(data.shape)
    data = data.sort_values(by='timestamp').tail(num_records)
    features = data[['voltage', 'current', 'temperature']]
    target = data['battery_health']
    if impute_strategy != 'None':
        if impute_strategy == 'Mean':
            imputer = SimpleImputer(strategy='mean')
        elif impute_strategy == 'Median':
            imputer = SimpleImputer(strategy='median')
        elif impute_strategy == 'Mode':
            imputer = SimpleImputer(strategy='most_frequent')
        features = imputer.fit_transform(features)
    else:
        features = features.dropna()
        target = target.loc[features.index]
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    return train_test_split(features, target, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_type):
    if os.path.exists(f"{''.join(selected_ml_model.lower().split())}_model.pkl"):
        model = pkl.load(open(f"{''.join(selected_ml_model.lower().split())}_model.pkl", 'rb'))
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(max_depth=20, random_state=0)
    elif model_type == 'Decision Tree':
        model = DecisionTreeRegressor()
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor()
    elif model_type == 'SVM':
        model = SVR()
    model.fit(X_train, y_train)
    return model

st.markdown("""
    <style>
        #ReportStatus { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.image('mtec.logo.png', caption='Engineering moves you to the future', width=250)

st.write('\n')

inputs_placeholder = st.empty()

uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"

model = None
model_data = None
submit_button = None
num_records = 10
with st.form(key='columns_in_form'):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("#### Battery Model Selection \n")
        models_df = retrieve_data(uri, 'vehicles', ['models'])
        model_options = models_df['model'].dropna().unique()
        selected_model = st.selectbox('Select Battery Model', model_options)
    with c2:
        st.write("#### Imputation Method \n")
        imputation_methods = ['None', 'Mean', 'Median', 'Mode']
        selected_imputation = st.selectbox('Select Imputation Method', imputation_methods)
    with c3:
        st.write("#### Length of set")
        num_records = st.number_input("Insert a Number", value=10, placeholder="Type a number...")
    with c4:
        st.write("#### Model Selection \n")
        ml_models = ['Linear Regression', 'SVM', 'Random Forest', 'Decision Tree', 'Gradient Boosting']
        selected_ml_model = st.selectbox('Select ML Model', ml_models)
    submit_button = st.form_submit_button(label = 'Train the Machine Learning Model')
    whole_df = retrieve_data(uri, 'vehicles', model_options)
    model_data = whole_df[whole_df['model'] == selected_model]

with inputs_placeholder.container():
    fig1, fig2, fig3 = st.columns(3)
    with fig1:
        st.write("Voltage Distribution")
        # st.write(f"{model_data.shape}")
        fig_voltage, ax_voltage = plt.subplots()
        hist = sns.histplot(model_data['voltage'], kde=True, ax=ax_voltage, color='#3EBABC', bins=10)
        ax_voltage.set_title('Voltage Distribution')
        fig_voltage.patch.set_facecolor('none')
        for bar in hist.patches:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() - 1
            if y > 0: 
                ax_voltage.text(x, y, f'{int(y)}', ha='center', va='bottom', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
        plt.grid(True)
        st.pyplot(fig_voltage)
    with fig2:
        st.write("Current Distribution")
        fig_current, ax_current = plt.subplots()
        hist = sns.histplot(model_data['current'], kde=True, ax=ax_current, color='#3EBABC', bins=10)
        ax_current.set_title('Current Distribution')
        fig_current.patch.set_facecolor('none')
        for bar in hist.patches:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() - 1
            if y > 0: 
                ax_current.text(x, y, f'{int(y)}', ha='center', va='bottom', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
        plt.grid(True)
        st.pyplot(fig_current)
    with fig3:
        st.write("Temperature Distribution")
        fig_temperature, ax_temperature = plt.subplots()
        hist = sns.histplot(model_data['temperature'], kde=True, ax=ax_temperature, color='#3EBABC', bins=10)
        ax_temperature.set_title('Temperature Distribution')
        fig_temperature.patch.set_facecolor('none')
        for bar in hist.patches:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() - 1
            if y > 0: 
                ax_temperature.text(x, y, f'{int(y)+1}', ha='center', va='bottom', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
        plt.grid(True)
        st.pyplot(fig_temperature)
    col1, col2 = st.columns(2)
    whole_df = whole_df.sort_values(by='timestamp')
    with col1:
        st.write('Whole Data')
        st.dataframe(whole_df.tail(num_records), hide_index=True)
    with col2:
        st.write('Filter Data')
        user_query = st.text_input('Enter your query (e.g., temperature > 30 and model == "Model 1"):', '')
        if user_query:
            try:
                filtered_df = whole_df.tail(num_records).query(user_query)
                st.dataframe(filtered_df, hide_index=True)
            except Exception as e:
                st.error(f"Error with query: {str(e)}")

if submit_button:
    st.session_state['fetch_counter'] += 1

    with st.status("Model Build Started...", expanded=True) as status:
        try:
            st.write("Setting up Environment...")
            time.sleep(2)
            st.write("Finalizing Training and Testing Data...")
            X_train, X_test, y_train, y_test = preprocess_data(model_data, selected_imputation, num_records)
            st.toast('Model Setup Complete!', icon='✅')
            time.sleep(1)
            st.write("Training Model...")
            model = train_model(X_train, y_train, selected_ml_model)
            st.toast('Model Training Complete!', icon='✅')
            time.sleep(1)
            st.write("Testing Model...")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.toast('Model Testing Complete!', icon='✅')
            time.sleep(1)
            st.write("Preparing Results...")
            time.sleep(3)
            status.update(label="Model Build Complete!", state="complete", expanded=False)
            st.toast('Model Build Complete!', icon='✅')
            with open(f"{''.join(selected_ml_model.lower().split())}_model.pkl", 'wb') as file:
                pkl.dump(model, file)
        except:
            st.toast('Model Build Failed!', icon='❗️')
    st.success(f'Model trained! MSE: {mse:.2f}')
        
    outputs_placeholder = st.empty()

    with outputs_placeholder.container():
        fig1, fig2, fig3 = st.columns(3)
        with fig1:
            st.write("Battery Data Monitor")
            temp = whole_df[whole_df['model'] == selected_model].tail(num_records)
            temp['battery_health_prediction'] = model.predict(temp[['voltage', 'current', 'temperature']].values)
            # st.write("Number of records in temp:", temp.shape[0])
            # st.write(temp[['voltage', 'current', 'temperature']].values)
            # st.write(X_test)
            data_df = pd.DataFrame({
                "Term": ['Voltage', 'Current', 'Temperature', 'Battery Health', 'Predicted Battery Health'],
                "Data": [
                    temp['voltage'].tolist(),
                    temp['current'].tolist(),
                    temp['temperature'].tolist(),
                    temp['battery_health'].tolist(),
                    temp['battery_health_prediction'].tolist(),
                ]
            })
            st.data_editor(
                data_df,
                column_config={
                    "name": "Term",
                    "Data": st.column_config.AreaChartColumn(
                        f"Terms (last {num_records} records)",
                        width="large",
                        help=f"The value of the terms in last {num_records} records.",
                    ),
                },
                hide_index=True,
            )
        with fig2:
            st.write("Battery Health Distribution")
            fig_battery_health, ax_battery_health = plt.subplots()
            hist = sns.histplot(model_data['battery_health'], kde=True, ax=ax_battery_health, color='#3EBABC', bins=10)
            ax_battery_health.set_title('Battery Health Distribution')
            fig_battery_health.patch.set_facecolor('none')
            for bar in hist.patches:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height() - 1
                if y > 0:
                    ax_battery_health.text(x, y, f'{int(y)}', ha='center', va='bottom', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
            plt.grid(True)
            st.pyplot(fig_battery_health)
        with fig3:
            st.write("Battery Health Results")
            fig_health, ax_health = plt.subplots()
            ax_health.plot(y_test.reset_index(drop=True), label='Actual Battery Health', marker='o', linestyle='-', color='#3EBABC')
            ax_health.plot(pd.Series(y_pred, index=y_test.index).reset_index(drop=True), label='Predicted Battery Health', marker='x', linestyle='--', color='red')
            ax_health.set_title('Comparison of Actual and Predicted Battery Health')
            ax_health.set_ylabel('Battery Health')
            ax_health.set_xlabel('Sample Index')
            ax_health.legend()
            ax_health.grid(True)
            fig_health.patch.set_facecolor('none')
            st.pyplot(fig_health)