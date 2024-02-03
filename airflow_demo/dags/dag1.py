from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pendulum
import os
import json
import iosense_connect as io
from Utilities.py_tools import *
import pandas as pd
import subprocess
import requests
import logging

api_key = os.environ.get("API")
url = os.environ.get("URL")
container_name = os.environ.get("CONTAINER_NAME")
iosense_data = io.DataAccess(api_key, url, container_name)

metadata_folder = "/opt/airflow/Database/Metadata"
schedule_folder = "/opt/airflow/Database/ScheduleMetadata"
parquet_dir = "/opt/airflow/Database/DagsData"
base_url = 'http://172.17.0.1:8000/'


# Read metadata
data_fetch_configs = {}
modelling_configs = {}

# Reading data_fetch metadata
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


# Reading modeling metadata
for filename in os.listdir(schedule_folder):
    if filename.endswith(".json"):
        with open(os.path.join(schedule_folder, filename), 'r') as file:
            metadata = json.load(file)
            config_key = metadata.get("fileName", filename.replace(".json", ""))
            modelling_data = metadata.get("modelling", {})
            scheduler_data = metadata.get("schedular", {})
            modelling_configs[config_key] = {
                "modelling": modelling_data,
                "start_date": scheduler_data.get("train_date", "") + " " + scheduler_data.get("train_time", ""),
                "frequency": scheduler_data.get("frequency", ""),
                "n_periods": scheduler_data.get("n_period", ""),
                "rolling": scheduler_data.get("rolling", "")
            }

def fetch_data_and_process(data_config_value):
    rolling = modelling_configs.get(data_config_value["file_name"], {}).get("rolling") == "True"
    start_time = pendulum.parse(data_config_value["start_time"])
    if rolling:
        start_time -= timedelta(days=int(data_config_value["period"]))
        # start_time=pendulum.parse(data_config_value["start_time"]) - timedelta(days=int(data_config_value["period"])),
    df = iosense_data.data_query(
        device_id=data_config_value["device_id"],
        sensors=data_config_value["sensors_list"],
        start_time=start_time,
        end_time=datetime.now(),
        cal=data_config_value["cal"],
        IST=data_config_value["ist"],
        db=data_config_value["gcs"]
    )

    # Here you should replace 'exec(task_code)' with a safer way to execute tasks
    for task_code in data_config_value["task"]:
        exec(task_code)

    return df

def data_modelling(model_config_value):
    modelling_data = model_config_value["modelling"]
    print(modelling_data)
    logging.info(modelling_data)
    algorithm = modelling_data.get("algorithm", "")
    if algorithm == 'Regression':
        route = 'regression'
    elif algorithm == 'Time-Series':
        models = modelling_data.get("models", [])
        if "LSTM" in models:
            selectedType = modelling_data.get("selectedType", "")
            route = 'custom_lstm' if selectedType == "Custom" else 'basic_lstm'
        else:
            route = 'timeseries'
    elif algorithm == 'Classification':
        route = 'classification'
    if route:
        response = requests.post(f"{base_url}{route}", json=modelling_data)
        return response.status_code, response.json() if response.ok else response.text
    else:
        return 404, None
      

def fetch_data_and_execute(config_key):
    subprocess.run(["chmod", "777", "/opt/airflow/Database/DagsData"])
    data_config_value = data_fetch_configs.get(config_key)
    if data_config_value:
        df = fetch_data_and_process(data_config_value)

        parquet_file_path = os.path.join(parquet_dir, f"{config_key}.parquet")
        if not os.path.exists(parquet_dir):
            os.makedirs(parquet_dir, exist_ok=True)
        df.to_parquet(parquet_file_path, index=False)
        print(f"Dataframe is saved as {parquet_file_path}")

        # df = pd.read_parquet(parquet_file_path)
        model_config_value = modelling_configs.get(config_key)
        if model_config_value:
            logging.info(f"Model config for {config_key}: {model_config_value}")
            try:
                response = data_modelling(model_config_value)
                logging.info(f"Modelling response: {response}")
            except Exception as e:
                logging.error(f"Error during modelling for {config_key}: {e}")



def generate_dag(config_key , config_value):
        dag_id = f'{config_key}'
        start_date_str = config_value['start_date']
        start_date = pendulum.from_format(start_date_str, "YYYY-MM-DD HH:mm:ss")
        default_args = {
            "owner": "airflow",
            "start_date": start_date,
            "max_active_runs": int(config_value["n_periods"]),
        }
        
        schedule_interval = None
        if config_value["frequency"] == "Weekly":
            schedule_interval = "@weekly"
        elif config_value["frequency"] == "Daily":
            schedule_interval = "@daily"
        elif config_value["frequency"] == "Hourly":
            schedule_interval = "@hourly"
        elif config_value["frequency"] == "Monthly":
            # This will schedule the job to run on the first day of every month
            schedule_interval = "0 0 1 * *"
        elif config_value["frequency"] == "Yearly":
            # This will schedule the job to run on the first day of every year
            schedule_interval = "0 0 1 1 *"

        dag = DAG(dag_id=dag_id, default_args=default_args, schedule_interval=schedule_interval)

        task = PythonOperator(
        task_id=f'execute_code_{config_key}',
        python_callable=fetch_data_and_execute,
        op_kwargs={'config_key': config_key},  
        provide_context=True,
        dag=dag,
        )

        return dag

# Iterate through the configs and generate a DAG for each entry
for config_key, config_value in modelling_configs.items():
    generated_dag = generate_dag(config_key, config_value)
    globals()[config_key] = generated_dag