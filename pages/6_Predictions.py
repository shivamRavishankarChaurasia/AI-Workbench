import requests
import pickle
import json
import os
import time
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import constants as c
import Utilities.py_tools as Manager
import Utilities.mappings as m
import plotly.graph_objects as go


def plot_graph_with_plotly(display_data, model_name):
    """
    Plot the forecasted values using Plotly.
    
    Parameters:
    - display_data: DataFrame containing 'Date' and 'PredictedValue' columns.
    - model_name: Name of the model for the plot title.
    """
    fig = go.Figure()

    
    for column in display_data.columns:
        fig.add_trace(go.Scatter(
            x=display_data.index,
            y=display_data[column],
            mode='lines+markers',
            name=column,
            line=dict(color='blue', width=2),  # Line color and width
            marker=dict(color='red', size=8)  
        ))

    fig.update_layout(
        title=f"{model_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted Value",
        showlegend=True,
        height=400,  # Adjust the height as needed
        width=800,
        font=dict(family="Arial", size=12),         
        margin=dict(l=50, r=50, t=50, b=50)  # Margins
    )

    return fig




st.set_page_config(layout="wide",page_title="Predictions",page_icon="https://storage.googleapis.com/ai-workbench/Prediction.svg")

Manager.faclon_logo()

st.subheader('Predictions')
tab = st.radio('Select tabs', ['Direct','Upload'], horizontal=True,  index=0, key='radio_key', label_visibility='collapsed')
st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)

if not os.path.exists('api_detailed.parquet'):
    bento_df = pd.DataFrame(columns=['File Name','run_id','model_name','timestamp'])
    bento_df.to_parquet('api_detailed.parquet',index=False)

prediction_df = pd.read_parquet('api_detailed.parquet')

if prediction_df.empty:
    st.error("No API was deployed")
else:
    if tab == "Direct":
        col1,col2 = st.columns([3.5,1.5])
        selected_api = col2.selectbox("Current APIs present",prediction_df['File Name'].to_list())

        display_df =prediction_df[prediction_df['File Name'] == selected_api]
        col2.dataframe(display_df.transpose(),use_container_width=True)

        exp_path = mlflow.get_experiment_by_name(selected_api).experiment_id
        run_path = display_df['run_id'].iloc[0]
        model_name = display_df['model_name'].iloc[0]

        placeholder = col1.empty()

        if model_name in m.CLASSIFICATION or model_name in m.REGRESSION:
            with open(f"mlruns/{exp_path}/{run_path}/display.pkl", 'rb') as pickle_file:
                fig = pickle.load(pickle_file)

            placeholder.info("To make predictions, put in values")
            df = col1.data_editor(fig,hide_index=True,use_container_width=True,height=200)
        else:
            with open(f"mlruns/{exp_path}/{run_path}/target_name.pkl", 'rb') as pickle_file:
                target_name = pickle.load(pickle_file)
            col_1 , col_2 = col1.columns([2,2])
            placeholder.info("Please select the number of Days to be forecasted.")
            no_of_days = col_1.number_input("No. of Days:", min_value=1 , label_visibility='collapsed')
            option = col_2.radio("Select Option:", ["Graph", "Dataframe"] , horizontal=True , label_visibility='collapsed')


        if col2.button('Predict', use_container_width=True, type='primary') or 'predict' in st.session_state:
            st.session_state['predict'] = 0
            
            if model_name in m.CLASSIFICATION or model_name in m.REGRESSION:
                missing_values = df['Values'].isnull() | (df['Values'] == '')

                if missing_values.any():
                    placeholder.error("There cannot be any null values")
                else:
                    placeholder.success("Applying Preprocessing")
                    try:
                        reportdf = pd.DataFrame()
                        for rows in range(0,len(df)):
                            if df.loc[rows,"Data Type"] in ['float64','int64']:
                                df.loc[rows,"Values"] = float(df.loc[rows,"Values"])
                                reportdf.loc[0,f"{df.loc[rows,'Column Name']}"] = float(df.loc[rows,"Values"])
                            else:
                                if df.loc[rows,"Values"] not in df.loc[rows,"Actual Values"]:
                                    raise ValueError(f"The Provided Value for {df.loc[rows,'Column Name']} is not present in Actual Values.")
                                else:
                                    reportdf.loc[0,f"{df.loc[rows,'Column Name']}"] = df.loc[rows,"Values"]

                        if model_name in m.REGRESSION:

                            reportdf_category = reportdf.select_dtypes(include='object')
                            reportdf_numerical = reportdf.drop(columns=reportdf.select_dtypes(include='object').columns)

                            with open(f"mlruns/{exp_path}/{run_path}/scaler.pkl", 'rb') as pickle_file:
                                scaler = pickle.load(pickle_file)

                            with open(f"mlruns/{exp_path}/{run_path}/encoder_x.pkl", 'rb') as pickle_file:
                                encoder_x = pickle.load(pickle_file)

                            encoded_data = encoder_x.transform(reportdf_category)
                            reportdf_category = pd.DataFrame(encoded_data.toarray(), columns=encoder_x.get_feature_names_out(reportdf_category.columns))
                            reportdf = pd.concat([reportdf_numerical, reportdf_category], axis=1)
                            prediction_data = scaler.transform(reportdf)

                        elif model_name in m.CLASSIFICATION:
                            reportdf_category = reportdf.select_dtypes(include='object')
                            reportdf_numerical = reportdf.select_dtypes(exclude='object')

                            with open(f"mlruns/{exp_path}/{run_path}/scaler.pkl", 'rb') as pickle_file:
                                scaler = pickle.load(pickle_file)

                            with open(f"mlruns/{exp_path}/{run_path}/encoder_x.pkl", 'rb') as pickle_file:
                                encoder_x = pickle.load(pickle_file)

                            for category in reportdf_category.columns:
                                encoder=encoder_x[category]
                                reportdf_category[category] = encoder.transform(reportdf_category[category])
                            
                            reportdf = pd.concat([reportdf_numerical, reportdf_category], axis=1)
                            prediction_data = scaler.transform(reportdf)

                        payload = {
                            "data": prediction_data.tolist(),
                            "run_id": f"{run_path}"
                        }      

                        response = requests.post("http://0.0.0.0:3000/classify",headers={"content-type": "application/json"},json=payload)       

                        if response.status_code == 200:
                            with open(f"mlruns/{exp_path}/{run_path}/target_name.pkl", 'rb') as pickle_file:
                                target_name = pickle.load(pickle_file)
                            
                            output = response.json()[0]

                            if model_name in m.CLASSIFICATION:
                                with open(f"mlruns/{exp_path}/{run_path}/encoder_y.pkl", 'rb') as pickle_file:
                                    encoder_y = pickle.load(pickle_file)
                                output = encoder_y.inverse_transform([int(output)])[0]

                            col1.success(f"""The Prediction {target_name} is {output}""")

                    except Exception as e:
                        col1.error(f"Exception: {e}")
            else:
                try:
                    payload = {
                            "data": no_of_days,
                            "run_id": f"{run_path}",
                            "modeltype": f"{model_name}"
                        }

                    if model_name in ["SARIMAX", "Prophet", "LSTM"]:
                        response = requests.post(
                            "http://0.0.0.0:3000/timeseries",
                            headers={"content-type": "application/json"},
                            json=payload
                        )
                                            
                    response_df = pd.DataFrame(response.json())
                    response_df['time'] = pd.to_datetime(response_df['time'], unit='ms')
                    response_df = response_df.set_index('time')
                    if option == "Graph":
                        plotly_fig = plot_graph_with_plotly(response_df, model_name)
                        col1.plotly_chart(plotly_fig)
                        placeholder.success("Forecasting Done Successfully")
                    else:
                        col1.dataframe(response_df.round(2).rename(columns={'PredictedValue': target_name}), use_container_width=True)
                        placeholder.success("Forecasting Done Successfully")
                    st.toast(f"Forecasting for {no_of_days} days")
                    time.sleep(10)
                    Manager.delete_in_page_session()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
        if col2.button('Delete', use_container_width=True, type='primary'):

            Manager.delete_api(experiment_name=selected_api)
            prediction_df.drop(prediction_df[prediction_df['File Name'] == selected_api].index[0],inplace=True)
            prediction_df.to_parquet('api_detailed.parquet',index=False)
            col2.success("Removed From Production!!")
            time.sleep(5)
            st.experimental_rerun()
            
    else:
        st.warning("Oopsstruction")
