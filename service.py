from bentoml.io import NumpyNdarray, PandasDataFrame
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
import pandas as pd
from pydantic import BaseModel
from bentoml.io import Multipart, NumpyNdarray, JSON
import numpy
import mlflow
import pickle
from Utilities.py_tools import get_experiment_id

"""
    BentoService serving a machine learning model loaded from MLflow.

    API Endpoint:
    classify(input_series: YourPydanticModel) -> np.ndarray
    timeseries(input_series: TimeSeries) -> pd.DataFrame:


    Parameters:
    - input_series: YourPydanticModel
        A Pydantic model specifying the input data structure.

"""
svc = bentoml.Service("Aiworkbench")

class features(BaseModel):
    data: list
    run_id: str
 
class TimeSeries(BaseModel):
    data: int
    run_id:str
    modeltype:str

@svc.api(input=JSON(pydantic_model=features), output=NumpyNdarray())
def classify(input_series: features) -> np.ndarray:
    """
        Make predictions using the loaded MLflow model.

        Parameters:
        - input_series: YourPydanticModel
            Input data for making predictions.

        Returns:
        - np.ndarray
            Predictions for the input data.
    """
    result=np.empty(0)
    predictor_model: mlflow.pyfunc.PyFuncModel = bentoml.mlflow.load_model(input_series.dict()["run_id"])
    print(predictor_model)
    for data in input_series.data:
        
        print(data)
        result = np.append(result,predictor_model.predict(np.reshape(np.array(data),(1,-1)).item()))
        print(result)
    return result


@svc.api(input=JSON(pydantic_model=TimeSeries), output=PandasDataFrame())
def timeseries(input_series: TimeSeries) -> pd.DataFrame:
    """
        Generate time series predictions based on the loaded MLflow model.

        Parameters:
        - input_series: YourTimeSeriesModel
            Input data for generating time series predictions.

        Returns:
        - pd.DataFrame
            Time series predictions.
    """
    prediction = []
    try:
        predictor_model: mlflow.pyfunc.PyFuncModel = bentoml.mlflow.load_model(input_series.dict()["run_id"])
        print(predictor_model)
        if input_series.modeltype == "SARIMAX":
            prediction = predictor_model.predict(input_series.data)
            prediction = prediction.rename('PredictedValue')
        elif input_series.modeltype == "Prophet":
            logged_model = f'runs:/{input_series.run_id}/Prophet'
            loaded_model = mlflow.sklearn.load_model(logged_model)
            future_data = loaded_model.make_future_dataframe(input_series.data)
            prediction = loaded_model.predict(future_data).reset_index().rename(columns={'ds': "time", 'yhat': 'PredictedValue'})[['time', 'PredictedValue']].tail(input_series.data)
        elif input_series.modeltype == "LSTM":
            forecast_values =[]
            logged_model = f'runs:/{input_series.run_id}/LSTM'
            loaded_model = mlflow.sklearn.load_model(logged_model)
            exp_path = get_experiment_id(input_series.run_id)
            with open(f"mlruns/{exp_path}/{input_series.run_id}/sequence_len.pkl", 'rb') as pickle_file:
                sequence_length = pickle.load(pickle_file)
            with open(f"mlruns/{exp_path}/{input_series.run_id}/target_name.pkl", 'rb') as pickle_file:
                target_name = pickle.load(pickle_file)
            with open(f"mlruns/{exp_path}/{input_series.run_id}/data_frame.pkl", 'rb') as pickle_file:
                df = pickle.load(pickle_file)
            last_sequence = df[target_name].tail(sequence_length)
            last_sequence_np = last_sequence.values.reshape((1, sequence_length, 1))
            for i in range(input_series.data):
                prediction = loaded_model.predict(last_sequence_np)[0, 0]
                forecast_values.append(prediction)
                last_sequence_np = np.append(last_sequence_np[:, 1:, :], [[[prediction]]], axis=1)
            last_date = last_sequence.index[-1] + pd.DateOffset(1)
            forecast_dates = pd.date_range(last_date, periods=input_series.data)
            prediction = pd.DataFrame({'index': forecast_dates, 'PredictedValue': forecast_values})
            prediction.set_index('index', inplace=True) 
        else:
            print(f"Unknown model type: {input_series.modeltype}")
    except Exception as e:
        print(f"Error: {e}")
        prediction = []
    print(pd.DataFrame(prediction))
    if input_series.modeltype == "Prophet":
        return pd.DataFrame(prediction)
    else:
        return pd.DataFrame(prediction).reset_index().rename(columns={"index": "time"})

      

