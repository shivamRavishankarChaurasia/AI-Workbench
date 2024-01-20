from datetime import datetime
import logging
import shutil
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    'depends_on_past': False,
    'email': "foo@email.com",
    'email_on_failure': True
}

MAX_LOG_DAYS = 30
LOG_DIR = '/efs/airflow/logs/'

def find_old_logs():
    # Query old dag runs and build the log file paths to be deleted
    # Example log directory looks like this:
    # '/path/to/logs/dag_name/task_name/2021-01-11T12:25:00+00:00'
    sql = f"""
        SELECT '{LOG_DIR}' || dag_id || '/' || task_id || '/' || replace(execution_date::text, ' ', 'T') || ':00' AS log_dir
        FROM task_instance
        WHERE execution_date::DATE <= now()::DATE - INTERVAL '{MAX_LOG_DAYS} days'
    """
    src_pg = PostgresHook(postgres_conn_id='airflow_db')
    conn = src_pg.get_conn()
    logging.info("Fetching old logs to purge...")
    with conn.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        logging.info(f"Found {len(rows)} log directories to delete...")
    for row in rows:
        delete_log_dir(row[0])


def delete_log_dir(log_dir):
    try:
        # Recursively delete the log directory and its log contents (e.g., 1.log, 2.log, etc)
        shutil.rmtree(log_dir)
        logging.info(f"Deleted directory and log contents: {log_dir}")
    except OSError as e:
        logging.info(f"Unable to delete: {e.filename} - {e.strerror}")

with DAG(
    dag_id="airflow_log_cleanup",
    start_date=datetime(2021, 1, 1),
    schedule_interval="00 00 * * *",
    default_args=default_args,
    max_active_runs=1,
    catchup=False,
) as dag:
    log_cleanup_op = PythonOperator(
        task_id="delete_old_logs",
        python_callable=find_old_logs
    )
