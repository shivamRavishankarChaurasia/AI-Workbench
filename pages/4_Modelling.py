import requests
import pickle
import json
import os
import time
import mlflow

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import constants as c
import plotly.graph_objects as go

import Utilities.py_tools as Manager
import Utilities.mappings as m
from statsmodels.tsa.stattools import adfuller

from BentoML.deployer import deploy_model

st.set_page_config(layout="wide",page_title="Modelling",page_icon="https://storage.googleapis.com/ai-workbench/Modeling.svg")

Manager.faclon_logo()

st.subheader('Modelling')
tab = st.radio('Select tabs', ['Train Models','Deployment' , 'Schedular'], horizontal=True,  index=0, key='radio_key', label_visibility='collapsed')
st.markdown("<hr style='margin:0px'>", unsafe_allow_html=True)
schedule_folder = "Database/ScheduleMetadata"

file_name = Manager.files_details()
iosense_true_experiments, iosense_false_experiments = Manager.get_mlflow_experiments()
if tab == 'Train Models':
    if len(file_name) > 1:
        # if 'iosense' not in st.session_state:
        #   st.session_state['iosense'] = False
        col1,col2 = st.columns([3.5,1.5])
        col1_1,col1_2 = col1.columns([1.5,3.5])
        col_1,col_2 = col1.columns([1.5,1.5])
        algorithm_select = col1_1.selectbox("Please Select Following Algorithm: ",['Regression','Classification',"Time-Series"])
        df = Manager.create_models_dataframe(algorithm_select)
        col1_2.write("    ")
        algo_df = col1_2.data_editor(df,hide_index=True,use_container_width=True)
        if True in algo_df['Check'].values:
            selected_algorithms = algo_df[algo_df['Check'].values]['Models'].to_list()
            with col2.container():
                selected_file = st.selectbox("Please select File:",file_name)
                is_file_scheduled = False
                # Check if the selected file is already scheduled
                for filename in os.listdir(schedule_folder):
                    if filename.endswith(".json"):
                        with open(os.path.join(schedule_folder, filename), 'r') as file:
                            metadata = json.load(file)
                            if metadata.get("fileName", "") == selected_file:
                                is_file_scheduled = True
                                break
                if  Manager.has_iosense_key(selected_file):
                    st.session_state['iosense'] = False
                else:
                    st.session_state['iosense'] = True

                df = Manager.read_parquet(file_name=selected_file)
                if len(df) > 0:
                    if algorithm_select == 'Regression':
                        selected_target = st.selectbox('Please Select Target:',df.columns)
                        verification = Manager.check_if_column_type(df,selected_target)
                        if verification ==  algorithm_select:
                            st.session_state['multi_select'] = False
                        else:
                            st.session_state['multi_select'] = True
                            col2.error(f"The Selected Target isnt {algorithm_select}")
                        remaining_columns = [col for col in df.columns if col != selected_target and not pd.api.types.is_datetime64_any_dtype(df[col])]
                        select_remaining_columns = st.multiselect("Please select X:",remaining_columns,disabled=st.session_state.multi_select)
                    elif algorithm_select == 'Time-Series':
                        try:
                            if len(selected_algorithms)>1:
                                col1.error("Please select only one algorithm")
                            select_remaining_columns = "0"

                            if len(df.select_dtypes(include=['datetime64','datetime64[ns]']).columns) != 0:
                                selected_time = st.selectbox('Please Select Time:', df.select_dtypes(include=['datetime64','datetime64[ns]']).columns)
                                if len(df.select_dtypes(include=np.number).columns) != 0:
                                    selected_target= st.selectbox('Please Select Target:', df.select_dtypes(include=np.number).columns)
                                    if 'SARIMAX' in selected_algorithms:
                                        p_value = adfuller(df[selected_target])[1].round(2)
                                        if p_value < 0.05:
                                            col1.info(f"The p-values: {p_value}. The data seems stationary.")
                                        else:
                                            col1.warning(f"The p-values: {p_value}. The data is not stationary. Affects model's performance.")
                                        select_remaining_columns = "yes"
                                        m = col2.number_input("Please provide the seasonality factor (m): ", min_value=0, max_value=20, help="Enter the number of lag observations included in the ARIMA model. Refer to the PACF plot to determine an appropriate value for 'p'." , key = "sarimax-m")

                                    elif 'Prophet' in selected_algorithms:
                                        m = "0"
                                        select_remaining_columns = "yes" 
                
                                    # elif 'LSTM' in selected_algorithms:
                                    else:
                                        max_sequence_length = int(0.8 * len(df))
                                        m = "0"
                                        selected_type = col2.selectbox("Please select the type  ",["Light" , "Medium" ,"Heavy" , "Custom"])
                                        if  selected_type == "Custom":
                                            lstm_layers = col2.number_input("Please provide the number of lstm layers", min_value=3, max_value=8 , value = 3 )
                                            lstm_units = col_2.number_input("Please provide the number of LSTM nodes:", min_value=1, max_value=100, value=32, help="The number of memory cells or neurons in the LSTM layer. This determines the complexity of the patterns the model can learn.")
                                            batch_size = col_1.number_input("Batch Size:", min_value=1, value= 32, help="The number of samples processed in each batch during training.")
                                            learning_rate = col_1.number_input("Please provide the learning rate:", min_value=0.00001, max_value=1.0, value=0.001, step=0.0001, format="%.5f", help="The step size during training. Controls how quickly or slowly the model learns.")
                                            sequence_length = col2.number_input("Sequence length", min_value=1, max_value=max_sequence_length, value=5, help="The number of time steps in each input sequence. This depends on how far back in time you want your model to consider for making predictions.")
                                            selected_activation = col_2.selectbox("Please select activation function:", ['relu', 'sigmoid', 'tanh'])
                                            select_remaining_columns = "yes"
                                        else:
                                           time_length = col2.number_input("Sequence length", min_value=1, max_value=50, value=3, help="The number of time steps in each input sequence. This depends on how far back in time you want your model to consider for making predictions.")       
                                           select_remaining_columns = "yes"
                                        
                                else:
                                    select_remaining_columns = "0"
                                    st.error('Numerical Target is Missing.')
                            else:
                                st.error('DateTime is mandatory.')
                                select_remaining_columns = "0"
                        except:
                            col2.error("Can't perform such operations")
                            select_remaining_columns = "0"

                    elif algorithm_select == 'Classification':
                        categorical_columns = df.select_dtypes(include='object').columns.to_list()
                        select_remaining_columns = "0"
                        if len(categorical_columns) > 0:
                            selected_target = st.selectbox("Please select the target:", categorical_columns)
                            verification = Manager.check_if_column_type(df, selected_target)

                            if verification == algorithm_select:
                                st.session_state['multi_select'] = False
                            else:
                                st.session_state['multi_select'] = True
                                col2.error(f"The Selected Target is not for {algorithm_select}")

                            remaining_columns = [col for col in df.columns if col != selected_target and not pd.api.types.is_datetime64_any_dtype(df[col])]
                            select_remaining_columns = st.multiselect("Please select X:", remaining_columns, disabled=st.session_state.multi_select)

                        else:
                            col2.warning("No categorical columns found.")

                    else:
                        select_remaining_columns = "0"


                    if len(select_remaining_columns) > 1:

                        with col1.container():
                            col_11,col_12 = st.columns(2)
                             
                            test_size = col_11.slider('Please Select Size of Test Data in Train-Test Split:', min_value=0.1, max_value=0.5, step=0.05)
                            scaling_disable= False

                            if algorithm_select == "Time-Series":
                              scaling_disable= True 
                            scaling_method = col_12.selectbox("Please select scaling:", ['Standard Scaler', 'Min Max Scaler', 'Robust Scaler'], disabled=scaling_disable)

                        if is_file_scheduled:
                            st.warning("The selected file is already scheduled. Scheduling and training are disabled.")
                            enable_scheduler = st.checkbox("Schedule", value=False, disabled=True, help="""To enable this feature, use Data Import --> Iosense Connect""")  
                            execute_train = st.button('Train your model', type='primary', use_container_width=True, disabled=True)
                        else:
                            enable_scheduler = st.checkbox("Schedule",disabled=st.session_state['iosense'], help="""To enable this feature, use Data Import --> Iosense Connect""")  
                            execute_train = st.button('Train your model', type='primary', use_container_width=True,disabled=enable_scheduler)

                        if enable_scheduler:
                            train_date, train_time, frequency, n_periods, rolling_checkbox = Manager.get_scheduling_parameters()
                            if st.button("Schedule", key="schedule_button", type="primary", use_container_width=True):
                                st.session_state.execute_train = True
                                st.success("Schedule Successfully")
                                
                        # if file_name 
                        if execute_train or 'execute_train'in st.session_state:
                            st.session_state.execute_train = True
                            try:
                                if algorithm_select == 'Regression':
                                    config = {
                                        "filename": selected_file,
                                        "algorithm": algorithm_select,
                                        "models": selected_algorithms,
                                        "yTarget": selected_target,
                                        "xTarget": select_remaining_columns,
                                        "testSize": test_size,
                                        "scaler": scaling_method,
                                        "iosense": enable_scheduler
                                    }
                                    if enable_scheduler:
                                        # generated_dag = generate_dag(config_key, config_value)
                                        Manager.create_scheduling_metadata(config ,train_date=train_date, train_time=train_time, n_periods=n_periods, frequency=frequency, rolling=rolling_checkbox )   
                                    else:
                                        response = requests.post('http://localhost:8000/regression',json=config)

                                elif algorithm_select == 'Time-Series':
                                    config = {
                                        "filename": selected_file,
                                        "algorithm": algorithm_select,
                                        "models": selected_algorithms,
                                        "yTarget": selected_target,
                                        "testSize": test_size,
                                        "timeCol": selected_time,
                                        "iosense":  enable_scheduler
                                    }
                                    if "LSTM" in selected_algorithms:
                                        if selected_type == "Custom":       
                                            config.update({
                                                "selectedType": selected_type,
                                                "activationFunction": selected_activation,
                                                "layers": lstm_layers,
                                                "batchSize": batch_size,
                                                "lstmNodes": lstm_units,
                                                "learningRate": learning_rate,
                                                "sequenceLength": sequence_length,
                                            })
                                            if enable_scheduler:
                                               Manager.create_scheduling_metadata(config ,train_date=train_date, train_time=train_time, n_periods=n_periods, frequency=frequency, rolling=rolling_checkbox )   
                                            else:  
                                                response = requests.post('http://localhost:8000/custom_lstm', json=config)
                                        else:
                                            config.update({
                                                "selectedType": selected_type,
                                                "sequenceLength": time_length
                                            })
                                            if enable_scheduler:
                                               Manager.create_scheduling_metadata(config ,train_date=train_date, train_time=train_time, n_periods=n_periods, frequency=frequency, rolling=rolling_checkbox )   
                                            else:  
                                                response = requests.post('http://localhost:8000/basic_lstm', json=config)
                                    else:
                                        config.update({
                                            "m": int(m)
                                        })
                                    
                                    if enable_scheduler:
                                       Manager.create_scheduling_metadata(config ,train_date=train_date, train_time=train_time, n_periods=n_periods, frequency=frequency, rolling=rolling_checkbox )   
                                    else:                                        
                                        response = requests.post('http://localhost:8000/timeseries', json=config)
                            
                                elif algorithm_select  == "Classification":
                                    config = {
                                        "filename": selected_file,
                                        "algorithm": algorithm_select,
                                        "models": selected_algorithms,
                                        "yTarget": selected_target,
                                        "xTarget": select_remaining_columns,
                                        "testSize": test_size,
                                        "scaler": scaling_method,
                                        "iosense": enable_scheduler
                                    }
                                    if enable_scheduler:
                                       Manager.create_scheduling_metadata(config ,train_date=train_date, train_time=train_time, n_periods=n_periods, frequency=frequency, rolling=rolling_checkbox )   
                                    else:
                                        response = requests.post('http://localhost:8000/classification',json=config)
                                else:
                                    response = 404
                                
                                if not enable_scheduler:
                                    if response.status_code == 200:
                                        st.toast('Training initiated')
                                        Manager.delete_in_page_session()
                                        time.sleep(10)
                                        st.experimental_rerun()
                                    else:
                                        col2.error("We were not able to initiate the training process")
                                        Manager.delete_in_page_session()
                                else:
                                    time.sleep(5)
                                    Manager.delete_in_page_session()
                                    st.experimental_rerun()
                                    
                            except Exception as e:
                                col2.error(f"We encountered an Error{e}")

                else:
                    st.error('Cant Perform the Operation due to missing data')
        else:
            col2.write("  ")
            col2.warning("Please select checkbox")

    else:
        st.error("Please Import CSV")

elif tab == "Deployment":
    if len(iosense_false_experiments) == 0:
        st.info('To Activate this Feature! Please Train the Models')
    else:
        col1,col2 = st.columns([3,2])
        placeholder = col1.empty()
        col1_1,col1_2 = col1.columns(2)
        selected_file = col2.selectbox("Please Select Trained Models",list(iosense_false_experiments))
        df = mlflow.search_runs(experiment_names=[selected_file])
        if len(df) > 0:
            display_df = df[['tags.Model','end_time','run_id']]
            display_df = display_df.rename(columns={'tags.Model':'Model Name',"end_time":'Time Created'})
            display_df['Time Created'] = display_df['Time Created'].apply(lambda x: pd.to_datetime(x).replace(microsecond=0).tz_localize(None) + pd.Timedelta(hours=5.5))
            display_df.insert(0, 'Check', False)
            display_df = col2.data_editor(display_df,hide_index=True,use_container_width=True,height=250)
            if (display_df['Check'] == True).any():
                if (display_df['Check'].sum() > 1):
                    col1.warning('Please select one check box at a time')
                else:
                    run_id = display_df[display_df['Check'] == True]['run_id'].iloc[0]
                    model_name = display_df[display_df['Check'] == True]['Model Name'].iloc[0]
                    try:
                        if df[df["run_id"] == run_id]["status"].iloc[0] == "FINISHED":

                            exp_id = df[df['run_id'] == run_id]['experiment_id'].iloc[0]

                            with open(f"mlruns/{exp_id}/{run_id}/figure.pkl", 'rb') as pickle_file:
                                fig = pickle.load(pickle_file)

                            with open(f"mlruns/{exp_id}/{run_id}/report.pkl", 'rb') as pickle_file:
                                metrices_df = pickle.load(pickle_file)
                            
                            placeholder.plotly_chart(fig,use_container_width=True)
                            
                            col2.dataframe(metrices_df,use_container_width=True,hide_index=True)

                            col2_1,col2_2 = col2.columns(2)

                            if col2_1.button('Deploy',use_container_width=True,type='primary'):
                                bento_response = deploy_model(experiment_name=selected_file,run_id=run_id)
                                bento_response = pd.DataFrame(bento_response)                                    

                                if not os.path.exists('api_detailed.parquet'):
                                    bento_df = pd.DataFrame(columns=['File Name','run_id','model_name','timestamp'])
                                    bento_df.to_parquet('api_detailed.parquet',index=False)

                                bento_df = pd.read_parquet('api_detailed.parquet')

                                if len(bento_df) == 0:
                                    bento_df = pd.concat([bento_df,bento_response])
                                elif bento_response['File Name'].iloc[0] in bento_df['File Name'].to_list():
                                    bento_df.loc[bento_df['File Name'] == bento_response['File Name'].iloc[0], 'run_id'] = bento_response['run_id'].iloc[0]
                                    bento_df.loc[bento_df['File Name'] == bento_response['File Name'].iloc[0], 'model_name'] = bento_response['model_name'].iloc[0]
                                    bento_df.loc[bento_df['File Name'] == bento_response['File Name'].iloc[0], 'timestamp'] = bento_response['timestamp'].iloc[0]
                                else:
                                    bento_df = pd.concat([bento_df,bento_response])

                                bento_df.to_parquet('api_detailed.parquet',index=False)
                                st.toast("Deployed API !! Please move to Prediction")

                            if col2_2.button('Delete', use_container_width=True, type='primary'):
                                bento_df = pd.read_parquet('api_detailed.parquet')

                                if run_id in bento_df['run_id'].to_list():
                                    col1.error("This Model is working in Production. Please delete it from API")
                                elif len(df) == 1:
                                    mlflow.delete_experiment(experiment_id=df['experiment_id'].iloc[0])
                                    st.experimental_rerun()
                                else:
                                    if model_name in m.TIMESERIES:
                                        os.remove(f"mlruns/{exp_id}/{run_id}/figure.pkl")
                                        os.remove(f"mlruns/{exp_id}/{run_id}/report.pkl")   
                                        os.remove(f"mlruns/{exp_id}/{run_id}/target_name.pkl")
                                    else: 
                                        os.remove(f"mlruns/{exp_id}/{run_id}/figure.pkl")
                                        os.remove(f"mlruns/{exp_id}/{run_id}/report.pkl")
                                        os.remove(f"mlruns/{exp_id}/{run_id}/scaler.pkl")
                                        os.remove(f"mlruns/{exp_id}/{run_id}/display.pkl")
                                        os.remove(f"mlruns/{exp_id}/{run_id}/target_name.pkl")
                                        os.remove(f"mlruns/{exp_id}/{run_id}/encoder_x.pkl")                                    
                                        if model_name in m.CLASSIFICATION:
                                            os.remove(f"mlruns/{exp_id}/{run_id}/encoder_y.pkl")                                                                    
                                        else:
                                            os.remove(f"mlruns/{exp_id}/{run_id}/target_name.pkl")

                                    mlflow.delete_run(run_id=f"{run_id}")
                                    st.experimental_rerun()
                        else:
                            col1.error("We've encountered an Error while generating Model...")
                    except Exception as e:
                        print("Exception occured in Modelling....Please Refresh!!",e)            

            else:
                col1.info("Please Select One Run ID!")
        else:
            col1.error("Fault Detected!!!!")



elif tab == "Schedular":
    # iosense_true_experiments = Manager.get_iosense_true_experiments()
    if len(iosense_true_experiments) == 0:
      st.info('No models are scheduled on iosense Data. Please schedule the modelling.')
    else:
        col1, col2 = st.columns([3, 2])
        placeholder = col1.empty()
        col1_1, col1_2 = col1.columns(2)
        selected_file = col2.selectbox("Please Select Trained Models", list(iosense_true_experiments))
        df = mlflow.search_runs(experiment_names=[selected_file])
        if len(df) > 0:
            display_df = df[['tags.Model','end_time','run_id']]
            display_df = display_df.rename(columns={'tags.Model':'Model Name',"end_time":'Time Created'})
            display_df['Time Created'] = display_df['Time Created'].apply(lambda x: pd.to_datetime(x).replace(microsecond=0).tz_localize(None) + pd.Timedelta(hours=5.5))
            display_df.insert(0, 'Schedule', False)
            display_df = col2.data_editor(display_df,hide_index=True,use_container_width=True,height=250)
            if (display_df['Schedule'] == True).any():
                if (display_df['Schedule'].sum() > 1):
                    col1.warning('Please select one check box at a time')
                else:
                    run_id = display_df[display_df['Schedule'] == True]['run_id'].iloc[0]
                    model_name = display_df[display_df['Schedule'] == True]['Model Name'].iloc[0]

                    try:
                        if df[df["run_id"] == run_id]["status"].iloc[0] == "FINISHED":

                            exp_id = df[df['run_id'] == run_id]['experiment_id'].iloc[0]

                            with open(f"mlruns/{exp_id}/{run_id}/figure.pkl", 'rb') as pickle_file:
                                fig = pickle.load(pickle_file)

                            with open(f"mlruns/{exp_id}/{run_id}/report.pkl", 'rb') as pickle_file:
                                metrices_df = pickle.load(pickle_file)
                            
                            placeholder.plotly_chart(fig,use_container_width=True)
                            
                            col2.dataframe(metrices_df,use_container_width=True,hide_index=True)

                            col2_1,col2_2 = col2.columns(2)

                            if col2_1.button('Deploy',use_container_width=True,type='primary'):
                                bento_response = deploy_model(experiment_name=selected_file,run_id=run_id)
                                bento_response = pd.DataFrame(bento_response)      

                            if col2_2.button('Delete', use_container_width=True, type='primary'):
                                json_file_path = f"Database/ScheduleMetadata/{selected_file}.json"
                                if os.path.exists(json_file_path):
                                    os.remove(json_file_path)
                                mlflow.delete_experiment(experiment_id=df['experiment_id'].iloc[0])
                                st.success("files and dags are deleted successfully.")
                                st.experimental_rerun()
                    except Exception as e:
                        print("Exception occured in Modelling....Please Refresh!!",e)   




