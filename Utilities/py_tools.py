import os
import time
import json
import glob
import math
import requests
import bentoml
import mlflow
import sklearn
import numpy as np
import constants as c
import pandas as pd
import streamlit as st
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import iosense_connect as io
import plotly.express as ex
import plotly.graph_objects as go
import pyparsing as pp
from statsmodels.tsa.stattools import adfuller
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from datetime import datetime
from celery.result import AsyncResult

import pendulum
from datetime import datetime, timedelta


mlflow.set_tracking_uri("http://172.17.0.1:5000")

"""
All Database Related Fuctionality
"""
def create_parquet(df: pd.DataFrame,file_name: str) -> bool:
    """Converts the csv into parquet

    Args:
        df (DataFrame): pd.dataframe
        file_name (_type_, optional): _description_. Contains the file name.
    """
    file_name = file_name.split(".")[0]
    st.toast(f"Initiated proccess to add {file_name} to Database")
    try:
        if os.path.exists(c.DEFAULT_STORAGE.format(file=file_name)) or len(df)==0:
            st.error(f"Uh-oh! We've got a name doppelganger: {file_name}")
        else:
            df.to_parquet(c.DEFAULT_STORAGE.format(file=file_name),index=False)
            create_metadata(file_name=file_name)
    except Exception as e:
        st.error(e)




def create_schedule_parquet(task_id, file_name, schedule_date, frequency, n_periods, schedule_time):
    try:
        parquet_path = c.DEFAULT_SCHEDULE_PATH.format(file="schedule_data")
        frequency_mapping = c.FREQUENCY_MAPPING.get(frequency.lower())
        if isinstance(frequency_mapping, timedelta):
            frequency_mapping_hours = frequency_mapping.total_seconds() / 3600
        else:
            frequency_mapping_hours = frequency_mapping
        delta_hours = frequency_mapping_hours * n_periods
        end_date = schedule_date + timedelta(hours=delta_hours)

        if os.path.exists(parquet_path):
            existing_schedule_df = pd.read_parquet(parquet_path)
        else:
            existing_schedule_df = pd.DataFrame(columns=["file_name", "task_id", "schedule_initialed","start_date","schedule_time","frequency", "end_date"])
            existing_schedule_df.to_parquet(parquet_path, index=False)

        new_schedule_row = pd.DataFrame({
            "file_name": [file_name],
            "task_id": [task_id],
            "schedule_initialed": [datetime.now()],
            "start_date": [schedule_date],
            "schedule_time": [schedule_time],
            "frequency": [frequency],
            "end_date": [end_date]
        })
        updated_schedule_df = pd.concat([existing_schedule_df, new_schedule_row], ignore_index=True)
        updated_schedule_df.to_parquet(parquet_path, index=False)
        st.toast("Schedule data is saved Successfully ")
    except Exception as e:
        print(f"An error occurred: {e}")



def update_parquet(df: pd.DataFrame,file_name: str):
    """Updates the existing parquet

    Args:
        df (DataFrame): pd.dataframe
        directory (_type_, optional): _description_. Defaults to c.default_storage_directory.
        file_name (_type_, optional): _description_. Contains the file name.
    """
    st.toast(f"Updating Database Table :{file_name}")
    df.to_parquet(c.DEFAULT_STORAGE.format(file=file_name), index=True)

def update_schedule_parquet(df: pd.DataFrame,file_name: str):
    """Updates the existing parquet

    Args:
        df (DataFrame): pd.dataframe
        directory (_type_, optional): _description_. Defaults to c.default_storage_directory.
        file_name (_type_, optional): _description_. Contains the file name.
    """
    st.toast(f"Updating Database Table :{file_name}")
    df.to_parquet(c.DEFAULT_SCHEDULE_PATH.format(file=file_name), index=True)


def read_parquet(file_name=None):
    """Takes in directory and file name to read parquet file

    Args:
        directory (Path, optional): _description_. Defaults to c.default_storage_directory.
        file_name (str, optional): _description_. Defaults to None.

    Returns:
        _type_: pd.DataFrame
    """
    result_df = pd.read_parquet(c.DEFAULT_STORAGE.format(file=file_name))
    return result_df


def read_schedule_parquet(file_name=None):
    """Takes in directory and file name to read parquet file

    Args:
        directory (Path, optional): _description_. Defaults to c.default_storage_directory.
        file_name (str, optional): _description_. Defaults to None.

    Returns:
        _type_: pd.DataFrame
    """
    parquet_path = c.DEFAULT_SCHEDULE_PATH.format(file=file_name)
    
    if os.path.exists(parquet_path):
        result_df = pd.read_parquet(parquet_path)
        return result_df
    else:
        print(f"The file {file_name} does not exist.")
        return None 


def files_details():
    """_summary_

    Returns:
        _type_: _description_
    """
    file_list = []
    files = glob.glob(c.DEFAULT_STORAGE.format(file="*"))
    for file in files:
        file_list.append(file.split("/")[-1].split(".")[0])
    return file_list


"""
All Related to Metadata
"""
def create_metadata(file_name: str):
    try:
        st.toast(f"Initialized Metadata for {file_name}")
        metadata = {
            'fileName': file_name,
            'timeCreated': str(datetime.now().replace(microsecond=0)),
            'timeModified': str(datetime.now().replace(microsecond=0)),
            'proccess': []
        }
        file_path = c.DEFAULT_METADATA.format(file=file_name)

        with open(file_path, "w") as json_file:
            json.dump(metadata, json_file)
        
        st.success("Metadata created Successfully")
        return True
    except Exception as e:
        st.error(f"Couldn't create Metadata. Error: {e}")
        return False
    


# this metadata is for iosense data
def invoke_iosense(file_name :str , Device_ID, Sensors, start_time, end_time, period, cal, db, ist):
    try:
        st.toast("Updating metadata")
        file_path = c.DEFAULT_METADATA.format(file=file_name)
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
        config['iosense']= {
            'Device_Id': Device_ID,
            'sensors': Sensors if Sensors is not None else None,
            'start_time': str(start_time),
            'end_time': str(end_time),
            'period': int(period),
            'cal': str(cal),
            'db': str(db),
            'IST': str(ist)
            }
        with open(file_path, "w") as json_file:
            json.dump(config, json_file)
        return True
    except Exception as e:
        st.error(f"Couldn't Modify iosense Metadata. Error: {e}")
        return False
        

def modify_metadata(file_name:str,new_process:list) -> bool:
    try:
        st.toast("Updating metadata")
        file_path = c.DEFAULT_METADATA.format(file=file_name)
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
        config['timeModified'] = str(datetime.now().replace(microsecond=0))
        for process in new_process:
            config['proccess'].append(process)
        with open(file_path, "w") as json_file:
            json.dump(config, json_file)
        st.toast("Metadata updated Successfully")
        return True
    except Exception as e:
        st.error(f"Couldn't Modify iosense Metadata. Error: {e}")
        return False


# def create_scheduling_metadata(config , train_date, train_time, n_periods, frequency, rolling):
#     try:
#         file_name = config["filename"]
#         st.toast(f"Initialized Metadata for {file_name}")
#         metadata = {
#             'fileName': file_name,
#             'modelling':config,
#             'schedular': {
#                 'train_date': str(train_date),
#                 'train_time': str(train_time),
#                 'n_period': int(n_periods),
#                 'frequency': str(frequency),
#                 'rolling': str(rolling)
#             }
#         }
#         file_path = c.DEFAULT_SCHEDULE_PATH.format(file=file_name)
#         with open(file_path, "w") as json_file:
#             json.dump(metadata, json_file, indent=4)

#         st.success("Metadata created Successfully")
#         return True
#     except Exception as e:
#         st.error(f"Couldn't create Metadata. Error: {e}")
#         return False
    

# this metadata is for iosense data
# def modify_scheduling_metadata(file_name: str, train_date, train_time, n_periods, frequency, rolling):
#     try:
#         st.toast("Updating metadata")
#         file_path = c.SCHEDULE_METADATA.format(file=file_name)
#         # file_path = f"Database/ScheduleMetadata/{file_name}.json"
#         with open(file_path, 'r') as config_file:
#             config = json.load(config_file)
#         config['schedular']= {
#             'train_date': str(train_date),
#             'train_time': str(train_time),
#             'n_period': int(n_periods),
#             'frequency': str(frequency),
#             'rolling': str(rolling)
#         }
#         with open(file_path, "w") as json_file:
#             json.dump(config, json_file)
#         return True
#     except Exception as e:
#         st.error(f"Couldn't Modify iosense Metadata. Error: {e}")
#         return False


def has_iosense_key(file_name: str) -> bool:
    file_path = c.DEFAULT_METADATA.format(file=file_name)
    try:
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
        # Check if 'iosense' key exists in the config dictionary
        return 'iosense' in config
        # return 'iosense' in config and config['iosense']
    except (FileNotFoundError, json.JSONDecodeError):
        # Return False in case of file not found or JSON decode error
        return False


# scheduling part part 
def get_scheduling_parameters():
    selected_date = st.date_input("Select the date")
    selected_time = st.time_input("Select the time")
    frequency = st.selectbox("Select the frequency", ["Hourly", "Daily", "Weekly", "Monthly", "Yearly"])
    periods = st.number_input("Number of times to retrain", min_value=1, max_value=8, value=1)
    rolling_checkbox = st.checkbox("Start_time:")
    return selected_date, selected_time, frequency, periods, rolling_checkbox



# Dashboarding 
# def save_dashboard_metadata(dash_board):


"""
Streamlit Sessions
"""
def faclon_logo():
    st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            background-image: url(https://storage.googleapis.com/ai-workbench/Deep%20sense.svg);
            background-repeat: no-repeat;
            padding-top: 0px;
            background-position: 26px 27px;
            background-size: 13%;
            position: relative; /* Add this line */
        }

        [data-testid="stSidebarNav"]::after {
            content: "";
            display: block;
            width: calc(100% - 40px); /* Reduce width by 40px total (20px from each side) */
            height: 1px; /* Thickness of the line */
            background-color: white; /* Color of the line */
            position: absolute;
            top: 85px; /* Adjust this value based on the position of the text and desired gap */
            left: 20px; /* Add left offset to move the line 20px from the left edge */
        }
    
        [data-testid="stSidebarNav"]::before {
            content: "AI Workbench";
            font-size: 35px;
            color: white; /* Set text color to white */
            position: absolute;
            top: 20px;
            right: 55px;
        }

        .st-emotion-cache-pkbazv{
            color: white;
        }

        .st-emotion-cache-vk3wp9{
            background: #2f55d4;
        }

        .st-emotion-cache-17lntkn{
            color: white;
        }

        .st-emotion-cache-z5fcl4 {
            padding: 2rem 2.5rem 10rem;
        }

        .st-emotion-cache-164nlkn ::after {
            content: "Made with ❤️ by Faclon Labs";
            display: block;
            position: absolute;
            bottom: 0;  
            left: 92%; 
            color: black; 
            font-size: 14px; 
            transform: translateX(-50%);  
            padding: 10px 0;
            text-align: center;
            background-color: #00000;
            width: 100%;
        }

        # header[data-testid="stHeader"]{
        #     display: none;
        # }
    </style>


    """,unsafe_allow_html=True,
    )

def delete_pages_sessions(key=None):
    """deletes pages session except the key value

    Args:
        key (_type_, optional): _description_. Defaults to None.
    """

    existing_session = st.session_state
    to_delete = []

    for session in st.session_state.keys():
        if session.startswith(key):
            continue
        to_delete.append(session)

    for delete in to_delete:
        del st.session_state[delete]

    print(to_delete,st.session_state)

def delete_in_page_session():
    """deletes all the sessions if called
    """

    st.session_state.clear()


"""
Data Import
"""

@st.cache_data
def verify_userid_iosense(user_key:str):
    io_sense = io.DataAccess(userid=user_key,
                        url=c.URL,
                        key=c.CONTAINER)
    values = io_sense.get_device_details()

    if type(values) == type(None):
        values = pd.DataFrame()
    return values


# @st.cache_data
def iosense_multi_select_concatinator(_io_sense,selected_device,select_sensors,start_time,end_time,db,cal,ist):

    df = pd.DataFrame()

    for device in selected_device:
        temp_df = _io_sense.data_query(device_id=device,
                                sensors=select_sensors,
                                start_time=str(start_time),
                                end_time=str(end_time),
                                cal=cal,
                                db=db,
                                IST=ist)
        
        temp_df.insert(0, 'Device ID', device)
        df = pd.concat([df,temp_df])

    return df


"""
Storage
"""

@st.cache_data  
def determine_categorial_columns(df,threshold):

    """Determines Category based on value counts

    Args:
        df (DataFrame): Data to find columns
        threshold (int): Ideally 5% cause 5% 0f 1000 is 50

    Returns:
        _type_: list
    """
    column_name = []
    for column in df.columns:
        if len(df[column].value_counts())/len(df) < threshold and len(df[column].value_counts()) > 1 and len(df[column].value_counts()) <= 30:
            column_name.append(column)
        
    return column_name

@st.cache_data
def get_table(df,tab0_categories):
    tab0_info = []
    for column in df.columns:
        information = {
        "Column Name": column,
        "dtype": df[column].dtype,
        "Count": df[column].count(),
        "Null Values %": df[column].isnull().mean().round(1) * 100,
        "Categorical": "Yes" if column in tab0_categories else "No"
        }
        tab0_info.append(information)

    return tab0_info

@st.cache_data
def concatinator(storage_concat):
    concat_df = pd.DataFrame()

    for concat_files in storage_concat:

        concat_df = pd.concat([concat_df,read_parquet(concat_files)])

    return concat_df

@st.cache_data
def get_merged_df(df1,df2,left_columns,right_columns,how):
    merge_df = pd.merge(df1,df2,
                    left_on=left_columns,right_on=right_columns,
                    how=how)

    return merge_df

@st.cache_data
def resampler(df,frequency,key,aggregation_mode):

    return df.resample(f'{frequency}{key}').agg(aggregation_mode)


def perform_individual_null_operation(df,agg_df):
    metadata = []
    for i in range(0,len(agg_df)):
        column = agg_df['Column'].iloc[i]
        method = agg_df['Aggregation'].iloc[i]
        key = agg_df['Key'].iloc[i]

        if key=="numeric":
            if method=='ffill' or method=='bfill':
                df[column] = df[column].fillna(method=f'{method}')
                metadata.append(f"df['{column}'].fillna(method='{method}')")
            elif method=='0':
                df[column] = df[column].fillna(0)
                metadata.append(f"df['{column}'].fillna(0)")
            elif method == 'Mean':
                df[column].fillna(df[column].mean(), inplace=True)
                metadata.append(f"df['{column}'].fillna(df['{column}'].mean())")
            elif method == 'Median':
                df[column].fillna(df[column].median(), inplace=True)
                metadata.append(f"df['{column}'].fillna(df['{column}'].median())")
            elif method == 'Min':
                df[column].fillna(df[column].min(), inplace=True)
                metadata.append(f"df['{column}'].fillna(df['{column}'].min())")
            elif method == 'Max':
                df[column].fillna(df[column].max(), inplace=True)
                metadata.append(f"df['{column}'].fillna(df['{column}'].max())")
        else:
            if method=='ffill' or method=='bfill':
                df[column] = df[column].fillna(method=f'{method}')
                metadata.append(f"df['{column}'].fillna(method='{method}')")
            elif method=='Max Frequency':
                df[column].fillna(df[column].value_counts().idxmax(),inplace=True)
                metadata.append(f"df['{column}'].fillna(df['{column}'].value_counts().idxmax())")
            elif method=='Min Frequency':
                df[column].fillna(df[column].value_counts().idxmin(),inplace=True)
                metadata.append(f"df['{column}'].fillna(df[{column}].value_counts().idxmin())")
    
    return df,metadata

"""
Data Preprocessing
"""


@st.cache_data
def get_trigonometric_value(df_col, function):
    
    if function == 'sin':
        return np.sin(df_col)
    elif function == 'cos':
        return np.cos(df_col)
    elif function == 'tan':
        return np.tan(df_col)
    elif function == 'arcsin':
        return np.arcsin(df_col)
    elif function == 'arccos':
        return np.arccos(df_col)
    elif function == 'arctan':
        return np.arctan(df_col)
    elif function == 'sinh':
        return np.sinh(df_col)
    elif function == 'cosh':
        return np.cosh(df_col)
    elif function == 'tanh':
        return np.tanh(df_col)
    else:
        return df_col




@st.cache_data
def get_outliers(df, col, method):
    """ To determine whether a given column (col) contains outliers, two methods are used (the Interquartile Range (IQR) method and the Z-Score method).
    IQR(Interquartile Range) calculates 75th percentile (Q3) and the 25th percentile (Q1) of the data and then subtract it and then find upper and lower bound
    In Z-Score we calculate mean and standard deviation of the selected column and then use the formula(x-mean/std) to detect the outliers 
    Args:
        df (_type_): _description_
        col (_type_): _description_
        method (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: pd.DataFrame()
    """

    if method == 'Inter Quantile Range':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    elif method == "Z-Score":
        mean = df[col].mean()
        std = df[col].std()
        z_scores = (df[col] - mean) / std
        df['Z-score'] = z_scores
        df_outliers = df[abs(z_scores) > 3]
    else:
        st.write("Select a valid outlier detection method ('IQR' or 'Z-score').")
    return df_outliers



@st.cache_data
def plot_outlier_graphs(outlier_df, col, method):
    """
    Generate a box plot to visualize outliers in a DataFrame column.

    Parameters:
        outlier_df (DataFrame): The DataFrame containing the data.
        col (str): The column name for which to create the box plot.
        method (str): The method used for detecting outliers.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure displaying the box plot.
    """
    fig = ex.box(outlier_df, x=col , points = 'outliers' , title='Box Plot to visualize Outliers' ,  template='plotly_dark' )
    return fig


@st.cache_data
def drop_outliers(df, outlier_df):
    """
Removes rows from the input DataFrame 'df' based on the indices provided in 'outlier_df'.

Parameters:
- df (DataFrame): The input DataFrame from which outliers will be removed.
- outlier_df (DataFrame): A DataFrame containing the indices of outliers to be removed.

Returns:
- DataFrame: A new DataFrame with the outliers removed.
"""
    df_new = df.drop(outlier_df.index)
    return df_new


@st.cache_data
def get_imbalance_cols(df):
    """The function get_imbalance_cols takes a DataFrame df and returns a list of columns with low cardinality, indicating imbalance column, using a default threshold of 10% of the DataFrame length
    Args:
        df (_type_): _description_

    Returns:
        _type_: List of imbalance column 
    """
    imb_col_lst = []
    cat_columns = df.select_dtypes(include="object").columns

    for col in cat_columns:
        unique_count = len(df[col].unique())
        if unique_count <= c.CARDINALITY_THRESHOLD:
    
            imb_col_lst.append(col)

    return imb_col_lst




    
@st.cache_data
def plot_imbalance_piechart(df, col):
    """
    Generate a pie chart to visualize data imbalance in a specified DataFrame column.

    Parameters:
        df (DataFrame): The DataFrame containing the data to be visualized.
        col (str): The name of the column to analyze for data imbalance.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure representing the pie chart.
    """
    fig = ex.pie(values=df[col].value_counts(), names=df[col].value_counts().keys().map(lambda x: f"{x[:8]}..." if len(x)>11 else x) )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(legend=dict(orientation='h', x=0, y=1.15), margin=dict(l=0, r=0, b=0, t=30))
    
    return fig



@st.cache_data
def smote_resampling(df, target_column):
    """
    Perform Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC) resampling.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_column (str): The target column name for resampling.

    Returns:
        pd.DataFrame: A resampled DataFrame with balanced classes.
    """
    smote_df = df.copy().dropna()
    X = smote_df.drop([target_column], axis=1)
    y = smote_df[target_column]

    # Identify categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=10)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    
    resampled_df = pd.concat([pd.DataFrame(y_resampled, columns=[target_column]), pd.DataFrame(X_resampled, columns=X.columns)], axis=1)
    
    return resampled_df


# @st.cache_data
# def smote_resampling(df, col):
#     """
#     Perform Synthetic Minority Over-sampling Technique (SMOTE) resampling.

#     Parameters:
#         df (pd.DataFrame): The input DataFrame containing the data.
#         col (str): The target column name for resampling.

#     Returns:
#         pd.DataFrame: A resampled DataFrame with balanced classes.
#     """
#     smote_df = df.copy()
#     smote_df = smote_df.dropna()
#     y = smote_df[col]
#     X = smote_df.drop([col], axis=1)

#     smote = SMOTE(random_state=10)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
    
#     resampled_df = pd.concat([pd.DataFrame(y_resampled, columns=[col]), pd.DataFrame(X_resampled, columns=X.columns)], axis=1)
    
#     return resampled_df





@st.cache_data
def over_under_sampling(df, advance_df, col):
    """By taking the data from advance_df that the user has provided, checks if oversampling (replacement = True) or 
    under-sampling (replacement = False) is needed based on sample size (use resample method of sklearn with the 
    replacement attribute) and then concatenates the resampled data with the remaining data to balance the classes.

    Args:
        df (pd.DataFrame): The original DataFrame.
        advance_df (pd.DataFrame): A DataFrame containing user input.
        col (str): The column to perform oversampling/undersampling on.

    Returns:
        pd.DataFrame: A DataFrame with balanced classes.
        list: Metadata information on the operations performed.
    """
    metadata = []  # Initialize metadata list to store information about operations
    resampled_samples = pd.DataFrame()  # Initialize the variable before the loop
    
    for index, row in advance_df.iterrows():
        class_name = None  # Initialize with a default value
        no_of_samples = 0  # Initialize with a default value
        
        if row['Select'] and row['No_of_Samples']:
            class_name = row['ClassName']
            no_of_samples = int(row['No_of_Samples'])
            
        if class_name is not None and no_of_samples > 0:
            class_samples = df[df[col] == class_name]
            remaining_df = df[df[col] != class_name]
            # metadata.append(f'class_samples = df[df["{col}"] == "{class_name}"]') 
            # metadata.append(f'remaining_df = df[df["{col}"] != "{class_name}"]') 

            if no_of_samples > class_samples.shape[0]:
                st.info("Please click the replacement button as the sample size is greater than the class count")
                if row['Replacement']:
                    resampled_samples = resample(class_samples, n_samples=no_of_samples, replace=True, random_state=10)
                    # metadata.append(f"resample({class_samples}, n_samples={no_of_samples}, replace=True, random_state=10)")
            elif no_of_samples < class_samples.shape[0]:
                resampled_samples = resample(class_samples, n_samples=no_of_samples, replace=False, random_state=10)
                # metadata.append(f"resample({class_samples}, n_samples={no_of_samples}, replace=False, random_state=10)")    
            else:
                print("Enter the data in advance_df")
            
            concatenated_df = pd.concat([remaining_df, resampled_samples])
            metadata.append(f'df = pd.concat([df[df["{col}"] != "{class_name}"], resample(df[df["{col}"] == "{class_name}"], n_samples={no_of_samples}, replace=True, random_state=10)])')
    return concatenated_df, metadata 


@st.cache_data
def perform_aggregation_Column(df, agg_df, group_by_col):
    """
    Perform column-wise aggregation on a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        agg_df (pd.DataFrame): A DataFrame specifying columns and aggregation functions.
        group_by_col (str): The column to group by.

    Returns:
        pd.DataFrame: The result of aggregation with renamed columns.
    """
    metadata = []
    agg_mapping = dict(zip(agg_df['Agg Column'], agg_df['Aggregation function']))
    grouped_df = df.groupby(group_by_col).agg(agg_mapping).reset_index()
    grouped_df.to_parquet("Test.parquet")
    new_column_names = [f"{col}_{agg}" for col, agg in grouped_df.columns]
    grouped_df.columns = new_column_names
    metadata.append(f"df = df.groupby('{group_by_col}').agg({dict(zip(agg_df['Agg Column'], agg_df['Aggregation function']))}).reset_index() ; df.columns = {new_column_names}")   
    return grouped_df , metadata 

    

@st.cache_data
def groupby_and_aggregate_DataFrame(combined_df , agg_func , df_col):
    """
    Group and aggregate a DataFrame based on a specific column.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be grouped and aggregated.
        agg_func (dict): A dictionary specifying aggregation functions for each column.
        df_col (str): The column name to group the DataFrame by.

    Returns:
        pd.DataFrame: The grouped and aggregated DataFrame.
    """
    
    grouped_df = combined_df.groupby(df_col).agg(agg_func)
    return grouped_df
   
# filtering 

@st.cache_data
def apply_filter_condition(df ,user_input, condition_dict):
    """
     Evaluate a user-provided condition against a DataFrame by using byparsing library of python 
     it handle  the parenthesis and logical operators on its own 

    Parameters:
        df (pd.DataFrame): The DataFrame to evaluate the condition against.
        user_input (str): The user-provided condition string with references to condition_dict.
        condition_dict (dict): A dictionary mapping references to conditions.

    Returns:
        pd.DataFrame or None: The filtered DataFrame if the condition is valid, or None if there's a parsing error.
    """
    metadata = []
    def replace_references(match):
        reference = match.group(0)
        if reference in condition_dict:
            return f"({condition_dict[reference]})"
        return reference

    # Replace references with conditions from the dictionary
    user_input = pp.re.sub(r'\b[cC]\d+\b', replace_references, user_input)
    print(user_input)
    df = df.query(user_input)
    metadata.append(f"""df = df.query("{user_input}")""")
    try:
        return df , metadata 
    except pd.errors.ParserError:
        return None 

    

@st.cache_data
def apply_encoding(df, encoding_col, encoding_type, class_list):
    """
    Apply data encoding to a DataFrame column.

    Parameters:
        df (pd.DataFrame): The DataFrame to apply encoding to.
        encoding_col (str): The column to be encoded.
        encoding_type (str): The type of encoding to apply ('Label encoding', 'Ordinal encoding', 'One-hot Encoding').
        class_list (list): A list of classes for ordinal encoding (ignored for other encoding types).

    Returns:
        pd.DataFrame, dict: The DataFrame with encoding applied and a label mapping dictionary.
    """ 
    if encoding_type == "Label encoding":
        le = sklearn.preprocessing.LabelEncoder()
        encoded_col = f"{encoding_col}_encoded"
        df[encoded_col] = le.fit_transform(df[encoding_col].values)
        label_map = dict(zip(df[encoding_col], df[encoded_col]))
       

    if encoding_type == "Ordinal encoding":
        encoded_dict = {}
        for i in class_list:
            encoded_dict[i] = len(class_list) - class_list.index(i)
        label_map = encoded_dict
        df[f"{encoding_col}_encoded"] = df[encoding_col].map(encoded_dict)
        

    if encoding_type == "One-hot Encoding":
        one_hot_encoded_df = pd.get_dummies(df, columns=[encoding_col], drop_first=True, dtype='int64')
        df = one_hot_encoded_df
        label_map = {}

    return df, label_map

@st.cache_data
def apply_scaling(df, scaling_type, scale_column, operation_type):
    """
    Apply data scaling to a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to apply scaling to.
        scaling_type (str): The type of scaling to apply ('Standard Scaling(Z-Score)', 'MinMax Scaling', 'Robust Scaling').

    Returns:
        pd.DataFrame: The DataFrame with scaling applied to numeric columns.
    """
    scaler = None  # Initialize the scaler variable outside the conditionals
    
    if scaling_type == 'Standard Scaling':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scaling_type == 'MinMax Scaling':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif scaling_type == 'Robust Scaling':
        scaler = sklearn.preprocessing.RobustScaler()

    if operation_type == "Single Column" and scale_column in df.columns:
        scaled_column = scaler.fit_transform(df[[scale_column]])  # Fit and transform the single column
        df[f"{scale_column}_scaled"] = scaled_column  # Add the scaled column

    elif operation_type == "Entire DataFrame":
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])  # Fit and transform all numeric columns

    return df

@st.cache_data
def apply_rolling(df, shift_roll_col, window_size, min_periods, rolling_functions, is_center):
    temp_df = df.copy()
    rolling_result = temp_df[shift_roll_col].rolling(window=window_size, min_periods=min_periods, center=is_center).agg(rolling_functions)
    rolling_result = rolling_result.to_frame()
    rolling_result.columns = [f'{col}_{rolling_functions}_rolling' for col in rolling_result.columns]
    df = pd.concat([df, rolling_result], axis=1)
    return df

# min-periods requires a set number of items to appear in order  to calculate the moving average 

@st.cache_data
def apply_shifting(df, shift_roll_col, shift_period):
    """
    Apply rolling computations to a DataFrame column.

    Parameters:
        df (pd.DataFrame): The DataFrame to apply rolling computations to.
        shift_roll_col (str): The column for rolling computations.
        window_size (int): The size of the rolling window.
        min_periods (int): The minimum number of periods required for each computation.
        rolling_functions (str or list): The rolling functions to apply (e.g., 'mean', ['mean', 'std']).
        is_center (bool): Whether to center the rolling window.

    Returns:
        pd.DataFrame: The DataFrame with rolling computations added as new columns.
    """
    df[f"{shift_roll_col}_shifted_{shift_period}"] = df[shift_roll_col].shift(shift_period)
    return df

@st.cache_data
def apply_arithmetic_operation(df, arithmetic_col, arithmetic_condition  , selected_functions=[], power_to=0 , log_function = None):
    """
    Apply arithmetic operations to a DataFrame based on user-defined conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to apply arithmetic operations to.
    - arithmetic_col (str): The name of the column to apply the operations on.
    - arithmetic_condition (str): The type of arithmetic operation ('trigonometric', 'power/root', 'log').
    - selected_functions (list): List of trigonometric functions to apply (for 'trigonometric' condition).
    - power_to (int): The power to raise the column values to (for 'power/root' condition).
    - log_function (str): The type of logarithmic function to apply ('loge' or 'log10' for 'log' condition).

    Returns:
    - pd.DataFrame: The DataFrame with arithmetic operations applied.
    """
 
    if arithmetic_condition == 'trigonometric':
        if len(selected_functions)>0:
            for val in selected_functions:
                df[f'{arithmetic_col}_{val}'] = get_trigonometric_value(df[arithmetic_col], val)

    elif arithmetic_condition == 'power/root':
        df[f"{arithmetic_col}^{power_to}"] = df[arithmetic_col].pow(power_to)

    elif arithmetic_condition == "logarithm":
            if log_function == "loge":
                df[f'{arithmetic_col}_loge'] = df[arithmetic_col].apply(lambda x: math.log(x) if x > 0 else np.nan)
            elif log_function == "log10":
                df[f'{arithmetic_col}_log10'] = df[arithmetic_col].apply(lambda x: np.log10(x) if x > 0 else np.nan)
            else:
              raise ValueError("Invalid log_function. Use 'loge' or 'log10'.")
    else:
        st.info("Select the operation to be performed")

    return df


@st.cache_data
def apply_logarithm_function(df, df_col, log_function):
    """These function calculate the logs value of the particular column df_col by using math inbuilt module of python and contact with the final df
    loge = natural logarithm 
    log10 = base 10 logarithm 

    Args:
        df (_type_): _description_
        df_col (_type_): _description_
        log_function (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: DataFrame df 
    """
    if log_function == "loge":
        df[f'{df_col}_loge'] = df[df_col].apply(lambda x: math.log(x) if x > 0 else np.nan)
    elif log_function == "log10":
        df[f'{df_col}_log10'] = df[df_col].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    else:
        raise ValueError("Invalid log_function. Use 'loge' or 'log10'.")
    
    return df 

@st.cache_data
def apply_arithematic_multicol_operation(df, formula_str, col_name):
    df[col_name] = eval(formula_str)
    return df

@st.cache_data
def apply_diff_function(df ,diff_col , period):
    """
    Calculate the difference between values in a DataFrame column.

    Parameters:
        df (pd.DataFrame): The DataFrame to calculate differences in.
        diff_col (str): The column to calculate differences for.
        period (int): The number of rows apart to calculate differences.

    Returns:
        pd.DataFrame: The DataFrame with difference values added as a new column.
    """
    # metadata = []
    df[f"{diff_col}_diff"]  = df[diff_col].diff(periods =period)

    return df 

"""
Data Exploration
"""

@st.cache_data
def corr_plot(data):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(data, cmap='viridis', ax=ax)
    return fig

@st.cache_data
def pairplot_fig(data, col):
    fig= sns.pairplot(data[col])
    return fig

@st.cache_data
def create_plotly_chart(chart_type, data, x_col, y_col):
    if chart_type == 'bar':
        fig = ex.bar(data, x=x_col, y=y_col)
    elif chart_type == 'line':
        fig = ex.line(data, x=x_col, y=y_col)
    elif chart_type == 'histogram':
        fig = ex.histogram(data, x=x_col, y=y_col)
    elif chart_type == 'area':
        fig = ex.area(data, x=x_col, y=y_col)

    return fig

@st.cache_data
def scatter_plot(data, x_col, y_col, color_col, size):
    fig = ex.scatter(data, x=x_col, y=y_col, color=color_col)
    fig.update_traces(marker_size=size)
    return fig

@st.cache_data
def pie_chart(data, value, name):
    fig = ex.pie(data,values= value, names=name)
    return fig

@st.cache_data
def time_series_plot(data, y_col, color):
    fig = ex.line(data, x=data.index, y=y_col)
    fig.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1, label="1y", step="year", stepmode="backward"),dict(count=2, label="3y", step="year", stepmode="backward"),dict(count=3, label="5y", step="year", stepmode="backward"),dict(step="all")])))
    fig.update_traces(line_color = color)
    return fig

@st.cache_data
def decomposition_plot(data, y_col, model):
    result = seasonal_decompose(data[y_col], model=model, period=12)
    res_fig = result.plot()
    return res_fig

@st.cache_data
def acf_plot(data, y_col, lags):
    fig, ax = plt.subplots(figsize=(12,4))
    plot_acf(data[y_col], lags=lags, ax=ax)
    plt.xlabel('Lag')
    return fig


@st.cache_data
def pacf_plot(data, y_col, lags):
    fig, ax = plt.subplots(figsize=(12,4))
    plot_pacf(data[y_col], lags=lags, ax=ax)
    plt.xlabel('Lag')
    return fig


@st.cache_data
def pca_plot(data, size, color):
    fig_pca = ex.scatter(data, x='PC1', y='PC2')
    fig_pca.update_traces(marker_size=size, marker_color= color)
    return fig_pca 

@st.cache_data
def tsne_plot(data, size, color):
    fig_tsne = ex.scatter(data, x='X', y='Y')
    fig_tsne.update_traces(marker_size=size, marker_color = color)
    return fig_tsne

@st.cache_data
def line_3d_plot(data, x_col, y_col, z_col):
    fig = ex.line_3d(data, x=x_col, y=y_col, z=z_col)
    return fig

@st.cache_data
def scatter_3d_plot(data, x_col, y_col, z_col, c_col, size):
  fig = ex.scatter_3d(data, x=x_col, y=y_col,z=z_col, color=c_col)                    
  fig.update_traces(marker_size=size)
  return fig


@st.cache_data
def surface_3d_plot(data, x_col, y_col, z_col):
    fig = go.Figure(data=[go.Surface(z=data[z_col], x=data[x_col], y=data[y_col])])
    return fig

"""
Modelling
"""
@st.cache_data
def create_models_dataframe(algo:str):

    if algo == 'Regression':
        df = pd.DataFrame({
            'Models': ['Linear Regression', 'Random Forest','Decision Tree', 'Support Vector Machine','XGBoost','Gradient Boosting','K-Nearest Neighbors' ,'SGDRegressor'],
        })

    elif algo == 'Classification':
        df = pd.DataFrame({
            'Models': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Naive Bayes', 'Support Vector Machine' ,'Gradient Boosting' ,'AdaBoost' , 'Multi-layer Perceptron classifier' ],
        })
    else:
        df = pd.DataFrame({
            'Models': ['SARIMAX' , 'Prophet' , 'LSTM'],
        })

    df['Check'] = False

    return df


@st.cache_data
def check_if_column_type(df: pd.DataFrame, column_name: str) -> str:
    
    # Get the dtype of the column
    col_type = df[column_name].dtype
    
    # Check if it's numeric
    if np.issubdtype(col_type, np.number):
        return 'Regression'
    
    # Check if it's datetime
    elif np.issubdtype(col_type, np.datetime64):
        return 'Time-Series'
    
    # Check if it's categorical
    elif col_type == 'object':
        return 'Classification'
    
    # If it's none of the above, return 'other'
    else:
        return 'other'
    


# def get_mlflow_experiments_name():
#     experiments = []
#     for experiment in mlflow.search_experiments():
#         name = experiment.name
#         if name == "Default":
#             continue
#         else:
#             experiments.append(name)
#     return experiments


def get_mlflow_experiments():
    experiments_with_iosense_true = []
    experiments_with_iosense_false = []

    for experiment in mlflow.search_experiments():
        name = experiment.name
        if name == "Default":
            continue

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Check if the 'tags.iosense' column exists and categorize accordingly
        if "tags.iosense" in runs.columns:
            if any(runs["tags.iosense"] == "True"):
                experiments_with_iosense_true.append(name)
            if any(runs["tags.iosense"] == "False"):
                experiments_with_iosense_false.append(name)
        else:
            experiments_with_iosense_false.append(name)

    return experiments_with_iosense_true, experiments_with_iosense_false




def delete_api(experiment_name):
    try:
        for x in bentoml.models.list():
            if bentoml.models.get(str(x.tag).split(":")[0]).info.labels["experiment_name"] == experiment_name:
                bentoml.models.delete(str(x.tag).split(":")[0])

    except Exception as e:
        print(f"Exception Occured while deleting the data: {experiment_name}")


import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix, class_labels, figsize=(8, 6)):
    """
    Plot a heatmap of a confusion matrix using Seaborn.

    Parameters:
    - confusion_matrix (array-like): The confusion matrix as a 2D array.
    - class_labels (list): List of class labels.
    - figsize (tuple): Figure size (width, height).

    Returns:
    - Matplotlib figure.
    """
    
    # Create a Seaborn heatmap
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)  # Adjust font size
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


@st.cache_data
def get_experiment_id(run_id):
    """
    Retrieve the experiment ID associated with a specific MLflow run.

    Parameters:
    - run_id (str): Unique identifier for the MLflow run.

    Returns:
    - str: Experiment ID corresponding to the provided run ID.
    """
    client = mlflow.tracking.MlflowClient()
    run_info = client.get_run(run_id)
    return run_info.info.experiment_id


@st.cache_data
def generate_schedule_dates(start_date, start_time, n_periods, frequency):
    """
    Generate a list of schedule dates based on the specified parameters.

    Parameters:
    - start_date (str): The starting date in the format 'YYYY-MM-DD'.
    - start_time (str): The starting time in the format 'HH:MM:SS'.
    - n_periods (int): Number of periods (dates) to generate.
    - frequency (str): Frequency of the schedule ('daily', 'weekly', etc.).

    Returns:
    - list: List of schedule dates as Pendulum datetime objects.
    """
    start_datetime = pendulum.parse(f"{start_date} {start_time}")
    interval = c.FREQUENCY_MAPPING.get(frequency.lower(), timedelta(days=1))
    schedule_dates = [start_datetime + i * interval for i in range(n_periods)]
    return schedule_dates



base_url = 'http://localhost:8000/'
iosense_data = io.DataAccess(c.API_KEY, c.URL, c.CONTAINER)
data_fetch_configs = {}
metadata_folder = c.DEFAULT_IOSENSE_METADATA


for filename in os.listdir(metadata_folder):
    if filename.endswith(".json"):
        with open(os.path.join(metadata_folder, filename), 'r') as file:
            metadata = json.load(file)
            if "iosense" in metadata:
                iosense_info = metadata.get("iosense", {})
                config_key = metadata.get("fileName", filename.replace(".json", ""))
                data_fetch_configs[config_key] = {
                    "file_name": config_key,
                    "device_id": iosense_info.get("Device_Id", ""),
                    "sensors_list": iosense_info.get("sensors", []),
                    "start_time": iosense_info.get("start_time", ""),
                    "end_time": iosense_info.get("end_time", ""),
                    "period": int(iosense_info.get("period", 0)),
                    "cal": iosense_info.get("cal", ""),
                    "ist": iosense_info.get("IST", ""),
                    "gcs": iosense_info.get("db", ""),
                    "task": metadata.get("proccess", [])
                }



@st.cache_data        
def fetch_data_and_process(data_config_value , rolling):
    start_time = pendulum.parse(data_config_value["start_time"])
    if rolling:
        start_time -= timedelta(days=int(data_config_value["period"]))
    df = iosense_data.data_query(
        device_id=data_config_value["device_id"],
        sensors=data_config_value["sensors_list"],
        start_time=start_time,
        end_time=datetime.now(),
        cal=data_config_value["cal"],
        IST=data_config_value["ist"],
        db=data_config_value["gcs"]
    )
    for task_code in data_config_value["task"]:
        exec(task_code)
    return df


@st.cache_data  
def data_modelling(config):
    algorithm = config.get("algorithm", "")
    if algorithm == 'Regression':
        route = 'regression'
    elif algorithm == 'Time-Series':
        models = config.get("models", [])
        if "LSTM" in models:
            selectedType = config.get("selectedType", "")
            route = 'custom_lstm' if selectedType == "Custom" else 'basic_lstm'
        else:
            route = 'timeseries'
    elif algorithm == 'Classification':
        route = 'classification'
    if route:
        response = requests.post(f"{base_url}{route}", json=config)
        return response.status_code, response.json() if response.ok else response.text
    else:
        return 404, None




@st.cache_data
def delete_tasks(task_ids):
    print("Hello WOrld")
    try:
        for task_id in task_ids:
            async_result = AsyncResult(task_id)
            if  async_result:
                async_result.revoke(terminate=True)
                st.toast(f"Task {task_id} revoked successfully.")
            else:
                print(f"Task {task_id} has already been executed, cannot revoke.")
    except Exception as e:
        print(f"Error deleting tasks: {e}")

