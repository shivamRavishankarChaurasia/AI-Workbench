# constant variables throughtout ADR
import os
import datetime
from dotenv import load_dotenv

load_dotenv(".env")
API_KEY = os.getenv("API")


"""
DEFAULT BASE FOR ENTIRE WORKBENCH
"""
DEFAULT_STORAGE = "Database/{file}.parquet"
DEFAULT_METADATA = "Database/Metadata/{file}.json"
SCHEDULE_METADATA = "Database/ScheduleMetadata/{file}.json"


URL = os.getenv('URL')
CONTAINER = os.getenv('CONTAINER_NAME')

BASE_DATE = datetime.datetime(1970, 1, 1)

"""
DATA PROCESSING MAPPING
"""
CARDINALITY_THRESHOLD = 10

NUMERIC_OPTIONS = ['mean', 'median', 'sum', 'min', 'max', 'count', 'std', 'var', 'first', 'last', 'nunique', 'size']
CATEGORICAL_OPTIONS = ['first', 'last', 'min', 'max', 'count', 'nunique', 'size']
ROLLING_OPTIONS = ['min', 'max', 'mean', 'median', 'sum', 'count', 'std', 'var']

FEATURE_ENGINEERING_OPTIONS = ['Encoding','Scaling','Shift/Roll Operation','Arithematic']
TRIGNOMETRIC_OPTIONS = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
