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

api_key = os.environ.get("API")
url = os.environ.get("URL")
container_name = os.environ.get("CONTAINER_NAME")
iosense_data = io.DataAccess(api_key, url, container_name)

metadata_folder = "/opt/airflow/Database/Metadata"
schedule_folder = "/opt/airflow/Database/scheduleMetadata"
# Read metadata
configs = {}
for filename in os.listdir(metadata_folder):
    if filename.endswith(".json"):
        with open(os.path.join(metadata_folder, filename), 'r') as file:
            metadata = json.load(file)
            if "iosense" in metadata:
                config_key = metadata.get("fileName", filename.replace(".json", ""))
                sensors_info = metadata.get("iosense", {})
                device_id = sensors_info.get("Device_Id", "")
                sensors_list = sensors_info.get("sensors", [])
                period = sensors_info.get("period", "")
                start_time = sensors_info.get("start_time", "")
                end_time = sensors_info.get("end_time", "")
                cal = sensors_info.get("cal", "")

                configs[config_key] = {
                    "task": metadata.get("proccess", []),
                    "start_date": metadata.get("train_date", "") + " " + metadata.get("train_time", ""),
                    "frequency": metadata.get("frequency", ""),
                    "n_periods": metadata.get("n_periods", ""),
                    "rolling": metadata.get("rolling", ""),
                    "device_id": device_id,
                    "period": period,
                    "sensors": sensors_list,
                    "start_time": start_time,
                    "end_time": end_time,
                    "cal": cal,
                }


# def fetch_data_and_execute(**kwargs):
#     subprocess.run(["chmod","777","/opt/airflow/    Database/DagsData"])
#     for config_key, config_value in configs.items():
#         if config_value["rolling"] == "True":
#             df = iosense_data.data_query(
#                 device_id=config_value["device_id"],
#                 sensors=config_value["sensors"],
#                 start_time=pendulum.parse(config_value["start_time"]) - timedelta(days=int(config_value["period"])),
#                 end_time=datetime.now(),
#                 cal=config_value["cal"],
#                 IST=True
#             )
#         else:
#             df = iosense_data.data_query(
#                 device_id=config_value["device_id"],
#                 sensors=config_value["sensors"],
#                 start_time=pendulum.parse(config_value["start_time"]),
#                 end_time=datetime.now(),
#                 cal=config_value["cal"],
#                 IST=True
#             )


#         for task_code in config_value["task"]:
#             exec(task_code)

#         parquet_dir = "/opt/airflow/Database/DagsData"
#         parquet_file_path = os.path.join(parquet_dir, f"{config_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}.parquet")

#         # Create the directory if it doesn't exist
#         if not os.path.exists(parquet_dir):
#             os.makedirs(parquet_dir , exist_ok=True)

#         # Save the DataFrame as a Parquet file
#         df.to_parquet(parquet_file_path)
#         print(f"Dataframe is saved as {parquet_file_path}")


def fetch_data_and_execute(**kwargs):
    subprocess.run(["chmod", "777", "/opt/airflow/Database/DagsData"])

    common_params = {
        "device_id": "",
        "sensors": [],
        "cal": "",
        "IST": True,
    }

    for config_key, config_value in configs.items():
        common_params["device_id"] = config_value["device_id"]
        common_params["sensors"] = config_value["sensors"]
        common_params["cal"] = config_value["cal"]

        if config_value["rolling"] == "True":
            common_params["start_time"] = pendulum.parse(config_value["start_time"]) - timedelta(days=int(config_value["period"]))
            common_params["end_time"] = datetime.now()
        else:
            common_params["start_time"] = pendulum.parse(config_value["start_time"])
            common_params["end_time"] = datetime.now()

        try:
            df = iosense_data.data_query(**common_params)

            for task_code in config_value["task"]:
                exec(task_code)
            parquet_dir = "/opt/airflow/Database/DagsData"
            parquet_file_path = os.path.join(parquet_dir, f"{config_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}.parquet")
            os.makedirs(parquet_dir, exist_ok=True)
            df.to_parquet(parquet_file_path)
            print(f"Dataframe is saved as {parquet_file_path}")

        except Exception as e:
            print(f"Error fetching and executing for {config_key}: {e}")


def generate_dag(config_key, config_value):
    dag_id = f'dynamic_dag_{config_key}'

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
        provide_context=True,
        dag=dag,
    )

    return dag


# Iterate through the configs and generate a DAG for each entry
for config_key, config_value in configs.items():
    generated_dag = generate_dag(config_key, config_value)
    globals()[config_key] = generated_dag
    # if "periods" in config_value:
    #     for period in range(int(config_value["periods"])):
    #         execution_date = pendulum.parse(config_value["start_date"]) + timedelta(days=period)
    #         generated_dag.run(start_date=execution_date, end_date=execution_date)

