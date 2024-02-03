from celery import Celery
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List
from Utilities.modelling import Regression, TimeSeries, Classification, LSTMTimeSeries
import os 
import Utilities.py_tools as Manager
import logging
import constants as c 
parquet_dir = "Database/DagsData"
app = FastAPI()

celery = Celery('tasks', broker='amqp://guest:guest@localhost:5672//')


class ReportFormat(BaseModel):
    filename: str
    algorithm: str
    models: List[str]
    yTarget: str
    xTarget: List[str]
    testSize: float
    scaler: str
    iosense:str


class TimeConfig(BaseModel):
    filename: str
    algorithm: str
    models: List[str]
    yTarget: str
    timeCol: str
    testSize: float
    m: int
    iosense:str

class CustomLstmConfig(BaseModel):
    filename: str
    algorithm: str
    models: List[str]
    yTarget: str
    testSize: float
    timeCol: str
    selectedType: str
    sequenceLength: int
    activationFunction: str
    layers: int
    batchSize: int
    lstmNodes: int
    learningRate: float
    iosense:str

class BasicLstmConfig(BaseModel):
    filename: str
    algorithm: str
    models: List[str]
    yTarget: str
    testSize: float
    timeCol: str
    selectedType: str
    sequenceLength: int
    iosense:str

class ResponseFormat(BaseModel):
    success: str


@celery.task
def generate_regression_model(config: dict, key: str):
    try:
        Regression(config=config, key=key).run()
    except Exception as e:
        print(f'{e}')


@celery.task
def generate_timeseries_model(config: dict, key: str):
    try:
        print(key)
        TimeSeries(config=config, key=key).run()
    except Exception as e:
        print(f'{e}')


@celery.task
def generate_classification_model(config: dict, key: str):
    try:
        Classification(config=config, key=key).run()
    except Exception as e:
        print(f'{e}')

@celery.task
def generate_custom_lstm_model(config: dict, key: str):
    try:
        LSTMTimeSeries(config=config, key=key).run()
    except Exception as e:
        print(f'{e}')

@celery.task
def generate_basic_lstm_model(config: dict, key: str):
    try:
        LSTMTimeSeries(config=config, key=key).run()
    except Exception as e:
        print(f'{e}')


@celery.task
def scheduled_task(config , config_key , rolling ):
    try:
        data_config_value = Manager.data_fetch_configs.get(config_key)
        if config_key:
            df = Manager.fetch_data_and_process(data_config_value , rolling)
            parquet_file_path = os.path.join(parquet_dir, f"{config_key}.parquet")
            if not os.path.exists(parquet_dir):
                os.makedirs(parquet_dir, exist_ok=True)
            df.to_parquet(parquet_file_path, index=False)
            logging.info(f"Dataframe is saved as {parquet_file_path}")
            if config:
                logging.info(f"Model config for {config_key}")
                response = Manager.data_modelling(config)
                logging.info(f"Modelling response: {response}")
    except Exception as e:
        logging.error(f"Error during processing for {config_key}: {e}")



def revoke_task(task_id):
    try:
        celery.control.revoke(task_id, terminate=True)
        print(f"Task {task_id} revoked successfully.")
    except Exception as e:
        print(f"Error revoking task {task_id}: {e}")


@app.post('/regression', response_model=ResponseFormat)
async def regression_model_call(request: ReportFormat):
    data = request.dict()
    print('Regression Call....', data)
    try:
        for model in data['models']:
            key = model
            generate_regression_model.delay(data, key)
        response = {"success": "true"}
    except Exception as e:
        print(e)
        response = {"success": "false"}
    return response


@app.post('/timeseries', response_model=ResponseFormat)
async def time_series_model(request: TimeConfig):
    data = request.dict()
    print('Timeseries Call....', data)
    try:
        for model in data['models']:
            key = model
            generate_timeseries_model.delay(data, key)
        response = {"success": "true"}
    except Exception as e:
        print(f"{e}")
        response = {"success": "false"}
    return response

@app.post('/classification', response_model=ResponseFormat)
async def classification_model_call(request: ReportFormat):
    data = request.dict()
    print('Classification Call....', data)
    try:
        for model in data['models']:
            key = model
            generate_classification_model.delay(data, key)
        response = {"success": "true"}
    except Exception as e:
        print(e)
        response = {"success": "false"}
    return response

@app.post('/custom_lstm', response_model=ResponseFormat)
async def lstm_model_call(request: CustomLstmConfig):
    data = request.dict()
    print('LSTM Model Call....', data)
    try:
        for model in data['models']:
            key = model
            generate_custom_lstm_model.delay(data, key)
        response = {"success": "true"}
    except Exception as e:
        print(e)
        response = {"success": "false"}
    return response

@app.post('/basic_lstm', response_model=ResponseFormat)
async def lstm_call(request: BasicLstmConfig):
    data = request.dict()
    print('LSTM Model Call....', data)
    try:
        for model in data['models']:
            key = model
            generate_basic_lstm_model.delay(data, key)
        response = {"success": "true"}
    except Exception as e:
        print(e)
        response = {"success": "false"}
    return response


