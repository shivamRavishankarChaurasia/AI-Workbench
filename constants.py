# constant variables throughtout ADR
import os
import datetime
from dotenv import load_dotenv
from datetime import datetime  as dt , timedelta 

load_dotenv(".env")
API_KEY = os.getenv("API")


"""
DEFAULT BASE FOR ENTIRE WORKBENCH
"""
DEFAULT_STORAGE = "Database/{file}.parquet"
DEFAULT_METADATA = "Database/Metadata/{file}.json"
DEFAULT_SCHEDULE_PATH = "Database/ScheduleData/{file}.parquet"
SCHEDULE_DATA = "Database/DagsData"
DEFAULT_IOSENSE_METADATA = "Database/Metadata"



URL = os.getenv('URL')
CONTAINER= b"WUHaM30tzELCSrfTypGF6A3KNubVIVHSiagTyZyWpUg="
BASE_DATE =  dt(1970, 1, 1)

"""
DATA PROCESSING MAPPING
"""
CARDINALITY_THRESHOLD = 10

NUMERIC_OPTIONS = ['mean', 'median', 'sum', 'min', 'max', 'count', 'std', 'var', 'first', 'last', 'nunique', 'size']
CATEGORICAL_OPTIONS = ['first', 'last', 'min', 'max', 'count', 'nunique', 'size']
ROLLING_OPTIONS = ['min', 'max', 'mean', 'median', 'sum', 'count', 'std', 'var']

FEATURE_ENGINEERING_OPTIONS = ['Encoding','Scaling','Shift/Roll Operation','Arithematic']
TRIGNOMETRIC_OPTIONS = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

FREQUENCY_MAPPING = {
    "daily": timedelta(days=1),
    "hourly": timedelta(hours=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),  # Assuming a month is approximately 30 days
    "yearly": timedelta(days=365),  # Assuming a year is approximately 365 days
}