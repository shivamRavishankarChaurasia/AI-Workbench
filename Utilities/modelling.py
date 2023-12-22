import os
import pickle
import mlflow
import mlflow.sklearn
import warnings
import traceback
import streamlit as st 
import numpy as np
import pandas as pd
import xgboost as xgb

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Regression-algos
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression , SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , roc_auc_score

# classification algos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score , precision_score, recall_score ,f1_score , confusion_matrix

# Time-series algos 
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from prophet import Prophet 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping  

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred): 
    """Provides mape based on actual values and predicted values

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_actual_vs_predicted_plotly(test_values, predicted_values, title="Actual vs Predicted Values"):
    """
    Plots true values and predicted values using Plotly.

    Parameters:
    - test_values (array-like): True values.
    - predicted_values (array-like): Predicted values from the model.
    - title (str): Plot title. Default is "Actual vs Predicted Values".
    
    Returns:
    - A Plotly Figure.
    """
    
    fig = go.Figure()

    # Line plot for true values
    fig.add_trace(go.Scatter(y=test_values, mode='lines', name='Actual', line=dict(color='blue')))

    # Line plot for predicted values
    fig.add_trace(go.Scatter(y=predicted_values, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))

    # Set titles and labels
    fig.update_layout(title=title,
                      yaxis_title='Value',
                      xaxis_title='Index',
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),
                      showlegend=True)

    return fig

def get_confusion_matrix(true_values, predicted_values):
    """
    Compute the confusion matrix for classification results.

    Parameters:
    - true_values (array-like): True class labels.
    - predicted_values (array-like): Predicted class labels.
    - labels (list): List of class labels to use in the confusion matrix. If None, it will be automatically determined from the data.

    Returns:
    - Confusion matrix as a 2D numpy array.
    """
    
    # Compute the confusion matrix
    cm = confusion_matrix(true_values, predicted_values)
    
    return cm


def plot_actual_vs_predicted_plotly_timeseries(test_values, predicted_values, title="Actual vs Predicted Values"):
    """
    Plots true values and predicted values using Plotly.

    Parameters:
    - test_values (array-like): True values.
    - predicted_values (array-like): Predicted values from the model.
    - title (str): Plot title. Default is "Actual vs Predicted Values".
    
    Returns:
    - A Plotly Figure.
    """
    
    fig = go.Figure()

    # Line plot for true values
    fig.add_trace(go.Scatter(x=test_values.index,y=test_values, mode='lines', name='Actual', line=dict(color='blue')))

    # Line plot for predicted values
    fig.add_trace(go.Scatter(x=test_values.index,y=predicted_values, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))

    # Set titles and labels
    fig.update_layout(title=title,
                      yaxis_title='Value',
                      xaxis_title='Index',
                      xaxis=dict(showgrid=True),
                      yaxis=dict(showgrid=True),
                      showlegend=True)

    return fig



def create_prediction_dataframe(fig):

    unique_values_dict = {
        "Column Name": [],
        "Data Type" : [],
        "Values":[],
        'Actual Values': []
    }

    for column in fig.columns.to_list():

        unique_values_dict["Column Name"].append(column)
        unique_values_dict["Data Type"].append(fig[column].dtype)
        unique_values_dict["Values"].append(None)

        if fig[column].dtype == 'object':
            unique_values_dict["Actual Values"].append(fig[column].unique())
        else:
            range_value = f"{fig[column].min()} - {fig[column].max()}"
            unique_values_dict["Actual Values"].append([f"{range_value}"])

    return pd.DataFrame(unique_values_dict)

def check_target_type(df):
    num_unique_classes = len(df.unique())
    print(num_unique_classes)
    if num_unique_classes == 2:
        return False
    else:
        return True
        


class Regression():
    def __init__(self, config, key):
        self.df = pd.read_parquet(f"""Database/{config['filename']}.parquet""")
        self.config = config
        self.key=key
        self.df = self.df.fillna(method='ffill')
        self.df = self.df.fillna(method='bfill')
        self.df_x = self.df[config['xTarget']]
        self.df_y = self.df[config['yTarget']]
        
        try:
            for category in self.df_x.columns:
                if np.issubdtype(self.df_x[category].dtype, np.datetime64):
                    self.df_x.drop([category], inplace=True, axis=1)

            self.backup = self.df_x

            self.encoder = OneHotEncoder(
                        categories='auto',  # Categories per feature
                        drop=None, # Whether to drop one of the features
                        sparse=True, # Will return sparse matrix if set True
                        handle_unknown='error' # Whether to raise an error 
                    ) 
            
            self.categorical_df = self.df_x.select_dtypes(include='object')
            self.numeric_df = self.df_x.drop(columns=self.df_x.select_dtypes(include='object').columns)
            
            encoded_data = self.encoder.fit_transform(self.categorical_df)
            self.encoded_df = pd.DataFrame(encoded_data.toarray(), columns=self.encoder.get_feature_names_out(self.categorical_df.columns))

            self.df_x = pd.concat([self.numeric_df, self.encoded_df], axis=1)
        except:
            self.df_x = self.df_x.select_dtypes(include=np.number)

        self.df_train_x , self.df_test_x, self.df_train_y , self.df_test_y = train_test_split(self.df_x , self.df_y ,test_size=config['testSize'], random_state = 42)

    def scaling_technique(self):
        if self.config['scaler'] == 'Standard Scaler':
            self.scaler = StandardScaler()
        elif self.config['scaler'] == 'Min Max Scaler':
            self.scaler = MinMaxScaler()
        elif self.config['scaler'] == "Robust Scaler":
            self.scaler = RobustScaler()

        self.df_train_x = self.scaler.fit_transform(self.df_train_x)
        self.df_test_x = self.scaler.fit_transform(self.df_test_x)

    def train_model(self):
        if self.key == 'Linear Regression':
            self.model = LinearRegression()
        elif self.key == 'XGBoost':
            self.model = XGBRegressor()
        elif self.key == 'Decision Tree':
            self.model = DecisionTreeRegressor()
        elif self.key == 'Random Forest':
            self.model = RandomForestRegressor()
        elif self.key == 'Gradient Boosting':
            self.model = GradientBoostingRegressor()
        elif self.key == 'K-Nearest Neighbors':
            self.model = KNeighborsRegressor()
        elif self.key == 'Support Vector Machine':
            self.model = SVR(kernel='linear')
        elif self.key == 'SGDRegressor':
            self.model = SGDRegressor()

        self.model.fit(self.df_train_x, self.df_train_y)
        self.valid_pred = self.model.predict(self.df_test_x)


    def evaluate_model(self):
        mae = mean_absolute_error(self.df_test_y, self.valid_pred)
        mse = mean_squared_error(self.df_test_y, self.valid_pred)
        rmse = mean_squared_error(self.df_test_y, self.valid_pred, squared=False)
        r2 = r2_score(self.df_test_y, self.valid_pred)
        mape = mean_absolute_percentage_error(self.df_test_y, self.valid_pred)

        self.metrics_dict = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

        self.performance_metrics_report = pd.DataFrame.from_dict({
            'Metric': list(self.metrics_dict.keys()),
            'Value': list(self.metrics_dict.values())
        })

        self.performance_metrics_report['Value'] = self.performance_metrics_report['Value'].round(2)

    def plot_graph(self):
        self.fig = plot_actual_vs_predicted_plotly(test_values=self.df_test_y, predicted_values=self.valid_pred)
        
    def log_mlflow(self):

        mlflow.set_experiment(self.config['filename'])

        with mlflow.start_run(run_name=self.key):
            mlflow.set_tag("Model", self.key)
            for key, value in self.metrics_dict.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(self.model, self.key)
            run_id = mlflow.active_run()

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/figure.pkl", "wb") as f:
                pickle.dump(self.fig, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/report.pkl", "wb") as f:
                pickle.dump(self.performance_metrics_report, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/encoder_x.pkl", "wb") as f:
                pickle.dump(self.encoder, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/display.pkl", "wb") as f:
                pickle.dump(create_prediction_dataframe(self.backup), f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/target_name.pkl", "wb") as f:
                pickle.dump(self.config['yTarget'], f)

    def run(self):
        self.scaling_technique()
        self.train_model()
        self.evaluate_model()
        self.plot_graph()
        self.log_mlflow()



class Classification():
    def __init__(self , config , key ):

        self.df = pd.read_parquet(f"""Database/{config['filename']}.parquet""")
        self.config = config 
        self.key = key
        self.df = self.df.fillna(method= 'ffill')
        self.df = self.df.fillna(method = 'bfill')
        self.df_x = self.df[config['xTarget']]
        self.df_y = self.df[config['yTarget']]

        self.multiclass = check_target_type(self.df_y)
        print(self.multiclass)

        try:
            for category in self.df_x.columns:
                if np.issubdtype(self.df_x[category].dtype, np.datetime64):
                    self.df_x.drop([category], inplace=True, axis=1)

            self.backup = self.df_x

            self.categorical_df = self.df_x.select_dtypes(include='object')
            self.numerical_df = self.df_x.select_dtypes(exclude='object')

            self.encoded_categorical_columns = {}
            for category in self.categorical_df.columns:
                self.encoder_x = LabelEncoder()
                encoded_category = self.encoder_x.fit_transform(self.categorical_df[category])
                self.encoded_categorical_columns[category] = self.encoder_x
                self.categorical_df[category] = encoded_category

            self.df_x = pd.concat([self.numerical_df, self.categorical_df], axis=1)
        except:
            self.df_x = self.df_x.select_dtypes(include=np.number)

        self.encoder_y = LabelEncoder()
        self.df_y = pd.Series(self.encoder_y.fit_transform(self.df_y), name=self.df_y.name)

        self.df_train_x, self.df_test_x, self.df_train_y, self.df_test_y = train_test_split(
            self.df_x, self.df_y, test_size=config['testSize'], random_state=42)
        
    def scaling_technique(self):
        if self.config['scaler'] == 'Standard Scaler':
            self.scaler = StandardScaler()
        elif self.config['scaler'] == 'Min Max Scaler':
            self.scaler = MinMaxScaler()
        elif self.config['scaler'] == "Robust Scaler":
            self.scaler = RobustScaler()

        self.df_train_x = self.scaler.fit_transform(self.df_train_x)
        self.df_test_x = self.scaler.fit_transform(self.df_test_x)

    def train_model(self, multiclass=True):
        if self.key == 'Logistic Regression':
            if multiclass:
                self.model = LogisticRegression(multi_class="ovr")  # Use 'ovr' for one-vs-rest
            else:
                self.model = LogisticRegression()
        # Add similar handling for other classifiers here
        elif self.key == 'Decision Tree':
            self.model = DecisionTreeClassifier()
        elif self.key == 'Random Forest':
            self.model = RandomForestClassifier()
        elif self.key == 'K-Nearest Neighbors':
            self.model = KNeighborsClassifier()
        elif self.key == 'Naive Bayes':
            self.model = GaussianNB()
        elif self.key == 'Support Vector Machine':
            if multiclass:
                self.model = SVC(decision_function_shape='ovr')  # Use 'ovr' for multiclass
            else:
                self.model = SVC()
        elif self.key == 'Gradient Boosting':
            self.model = GradientBoostingClassifier()
        elif self.key == 'AdaBoost':
            self.model = AdaBoostClassifier()
        elif self.key == 'Multi-layer Perceptron classifier':
            self.model = MLPClassifier()

        self.model.fit(self.df_train_x, self.df_train_y)
        self.valid_pred = self.model.predict(self.df_test_x)

    def evaluate_model(self):
        accuracy = accuracy_score(self.df_test_y, self.valid_pred)
        precision = precision_score(self.df_test_y, self.valid_pred, average='weighted')  # Change 'average' to 'weighted'
        recall = recall_score(self.df_test_y, self.valid_pred, average='weighted')  # Change 'average' to 'weighted'
        f1 = f1_score(self.df_test_y, self.valid_pred, average='weighted')  # Change 'average' to 'weighted'
        # roc_auc = roc_auc_score(self.df_test_y, self.valid_pred, average='weighted',multi_class="ovr")  # Change 'average' to 'weighted'

        self.metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
            # 'ROC AUC': roc_auc
        }

        self.performance_metrics_report = pd.DataFrame.from_dict({
            'Metric': list(self.metrics_dict.keys()),
            'Value': list(self.metrics_dict.values())
        })

        self.performance_metrics_report['Value'] = self.performance_metrics_report['Value'].round(2)


    def plot_graph(self):
        self.true_values = self.encoder_y.inverse_transform(self.df_test_y)
        self.predicted_values = self.encoder_y.inverse_transform(self.valid_pred)
        self.fig = get_confusion_matrix(true_values=self.true_values , predicted_values=self.predicted_values)

        class_labels = self.encoder_y.classes_

        self.fig = go.Figure(data=go.Heatmap(
            z=self.fig,
            x=class_labels,
            y=class_labels,
            colorscale='Spectral',  # You can choose a different color scale
            colorbar=dict(title='Count')
        ))

        # Customize the layout
        self.fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
        )
        
    def log_mlflow(self):

        mlflow.set_experiment(self.config['filename'])

        with mlflow.start_run(run_name=self.key):
            mlflow.set_tag("Model", self.key)
            for key, value in self.metrics_dict.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(self.model, self.key)
            run_id = mlflow.active_run()

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/figure.pkl", "wb") as f:
            pickle.dump(self.fig, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/report.pkl", "wb") as f:
            pickle.dump(self.performance_metrics_report, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/encoder_y.pkl","wb") as f:
            pickle.dump(self.encoder_y,f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/encoder_x.pkl","wb") as f:
            pickle.dump(self.encoded_categorical_columns,f)
        
        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/display.pkl", "wb") as f:
            pickle.dump(create_prediction_dataframe(self.backup), f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/target_name.pkl", "wb") as f:
            pickle.dump(self.config['yTarget'], f)

    def run(self):
        self.scaling_technique()
        self.train_model(multiclass=self.multiclass)
        self.evaluate_model()
        self.plot_graph()
        self.log_mlflow()
        traceback.print_exc()



class TimeSeries():
    def __init__(self, config, key):
        self.df = pd.read_parquet(f"""Database/{config['filename']}.parquet""")
        self.config = config
        self.key = key
        self.df[config['timeCol']] = pd.to_datetime(self.df[config['timeCol']])
        self.df.sort_values(by=config['timeCol'], inplace=True)
        self.df.set_index(config['timeCol'], inplace=True)
        self.df = self.df.fillna(method='ffill')
        self.df = self.df.fillna(method='bfill')
        self.df_y = self.df[config['yTarget']]      
        self.df_y = self.df_y.rename_axis("DATE")
        self.df_y.dropna(axis=0, inplace=True)   
        self.split_index = int(len(self.df_y) * (1 - config['testSize']))
        self.train_y = self.df_y.iloc[:self.split_index]
        self.test_y = self.df_y.iloc[self.split_index:]

    def train_models(self):
        if self.key == 'SARIMAX':
            self.arima_model()
        elif self.key  == "Prophet":
            self.prophet_model()
        

    def arima_model(self):
        if self.config['m'] == 0:
            self.fitted_model = auto_arima(
                y=self.train_y,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
                seasonal=False
            )
        else:
            self.fitted_model = auto_arima(
                y=self.train_y,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
                seasonal=True,
                m=self.config['m']
            )
        self.valid_pred = self.fitted_model.predict(n_periods=len(self.test_y))
        # self.model = ARIMA(self.train_y, order=(self.config['p'],self.config['d'],self.config['q']))
        # self.fitted_model = self.model.fit()        
        # self.valid_pred = self.fitted_model.forecast(steps=len(self.test_y))
       
    def prophet_model(self):
            self.df_prophet_train = pd.DataFrame({'ds': self.train_y.index, 'y': self.train_y.values})
            self.df_prophet_train.set_index('ds', inplace=True)
            self.df_prophet_test = pd.DataFrame({'ds': self.test_y.index, 'y': self.test_y.values})
            self.df_prophet_test.set_index('ds', inplace=True)
          
            self.df_prophet_train['ds'] = pd.to_datetime(self.df_prophet_train.index)
            self.df_prophet_test['ds'] = pd.to_datetime(self.df_prophet_test.index)
          
            # Calculate the time difference between the first two dates in training data
            time_diff = (self.df_prophet_train.index[1] - self.df_prophet_train.index[0]).total_seconds()
    
            # Determine frequency based on time difference
            if time_diff <= 24 * 60 * 60:  # Less than or equal to 1 day
               self.frequency = 'D'  # Daily
            elif time_diff <= 7 * 24 * 60 * 60:  # Less than or equal to 1 week
               self.frequency = 'W'  # Weekly
            elif time_diff <= 31 * 24 * 60 * 60:  # Less than or equal to 1 month
               self.frequency = 'M'  # Monthly
            elif time_diff <= 365 * 24 * 60 * 60:  # Less than or equal to 1 year
               self.frequency = 'Y'  # Yearly
            elif time_diff <= 60 * 60:  # Less than or equal to 1 hour
               self.frequency = 'H'  # Hourly
            elif time_diff <= 31 * 24 * 60 * 60:  # Less than or equal to 1 month
               self.frequency = 'MS'  # Month start
            elif time_diff <= 60:  # Less than or equal to 1 minute
               self.frequency = 'T'  # Minutely
            elif time_diff <= 1:  # Less than or equal to 1 minute
               self.frequency = 'S'  # Secondly
            else:
               self.frequency = 'D'  # Default to Daily
                                
            model = Prophet()
            self.fitted_model = model.fit(self.df_prophet_train)
            future = self.fitted_model.make_future_dataframe(periods=len(self.df_prophet_test) , freq = self.frequency )
            forecast = self.fitted_model.predict(future)
            self.valid_pred_df = forecast.tail(len(self.df_prophet_test))
            self.valid_pred_df.set_index('ds', inplace=True)
            self.valid_pred = self.valid_pred_df['yhat']


    def evaluate_model(self):
        mae = mean_absolute_error(self.test_y.values, self.valid_pred.values)
        mse = mean_squared_error(self.test_y.values, self.valid_pred.values)
        r2 = r2_score(self.test_y.values, self.valid_pred.values)
        mape = mean_absolute_percentage_error(self.test_y.values, self.valid_pred.values)

        self.metrics_dict = {
            'Mean Absolute Error (MAE)': mae,
            'Mean Squared Error (MSE)': mse,
            'R-squared (R^2)': r2,
            'Mean Absolute Percentage Error (MAPE)': mape
        }

        self.performance_metrics_report = pd.DataFrame.from_dict({
            'Metric': list(self.metrics_dict.keys()),
            'Value': list(self.metrics_dict.values())
        })
        self.performance_metrics_report['Value'] = self.performance_metrics_report['Value'].round(2)

    def plot_graph(self):
        self.fig = plot_actual_vs_predicted_plotly_timeseries(test_values=self.test_y, predicted_values=self.valid_pred)
    
    def log_mlflow(self):

        mlflow.set_experiment(self.config['filename'])

        with mlflow.start_run(run_name=self.key):
            mlflow.set_tag("Model", self.key)
            run_id = mlflow.active_run()
            if self.key == 'SARIMAX':
                mlflow.sklearn.log_model(self.fitted_model, self.key)
            elif self.key == 'Prophet':
                mlflow.sklearn.log_model(self.fitted_model, self.key)
                with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/freq.pkl", "wb") as f:
                    pickle.dump(self.frequency, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/model.pkl", "wb") as f:
            pickle.dump(self.fitted_model, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/figure.pkl", "wb") as f:
            pickle.dump(self.fig, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/report.pkl", "wb") as f:
            pickle.dump(self.performance_metrics_report, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/target_name.pkl", "wb") as f:
            pickle.dump(self.config['yTarget'], f)

    def run(self):
        self.train_models()
        self.evaluate_model()
        self.plot_graph()
        self.log_mlflow()


class LSTMTimeSeries():
    def __init__(self, config , key):
        self.df = pd.read_parquet(f"Database/{config['filename']}.parquet")
        self.config = config
        self.key = key
        self.df = self.df[[config['timeCol'], config['yTarget']]]   
        self.df[config['timeCol']] = pd.to_datetime(self.df[config['timeCol']])
        self.df.sort_values(by=config['timeCol'], inplace=True)
        self.df.set_index(config['timeCol'], inplace=True)
        self.df = self.df.fillna(method='ffill')
        self.df = self.df.fillna(method='bfill')
        self.df_y = self.df[config['yTarget']]
        self.df_y = self.df_y.rename_axis("DATE")        
        self.df_y.dropna(axis=0, inplace=True)

        self.split_index = int(len(self.df_y) * (1 - config['testSize']))
        self.df_train_y = self.df_y.iloc[:self.split_index]
        self.df_test_y = self.df_y.iloc[self.split_index:]
   
    def lstm_model(self):
        if self.config['selectedType'] in ["Light", "Medium", "Heavy"]:
            if self.config['selectedType'] == 'Light':
                num_lstm_layers = 3
                lstm_nodes_list = [64, 32, 16] 
                batch_size = 32
            elif self.config['selectedType'] == 'Medium':
                num_lstm_layers = 5
                lstm_nodes_list = [128, 64, 32, 16, 8] 
                batch_size = 64
            elif self.config['selectedType'] == 'Heavy':   
                num_lstm_layers = 6
                lstm_nodes_list = [256, 128, 64, 32, 16, 8] 
                batch_size = 128
            else:
                num_lstm_layers = 4
                lstm_nodes_list = [128, 64, 32, 16] 
                batch_size = 64

        #   Adjust sequence length based on the selected type
            self.sequence_len = self.config['sequenceLength']
            self.train_generator = TimeseriesGenerator(self.df_train_y.values, targets=self.df_train_y.values, length=self.sequence_len, batch_size=batch_size)
            self.test_generator = TimeseriesGenerator(self.df_test_y.values  , targets = self.df_test_y.values, length=self.sequence_len, batch_size=batch_size)
            self.model = Sequential()  
            for i in range(num_lstm_layers - 1):
               self.model.add(LSTM(units=lstm_nodes_list[i], activation='relu', return_sequences=True, input_shape=(self.sequence_len, 1)))
            self.model.add(LSTM(units=lstm_nodes_list[-1], activation='relu'))
            self.model.add(Dense(units=50, activation='relu'))
            self.model.add(Dense(units=25, activation='relu'))
            self.model.add(Dense(units=1))

            optimizer = Adam(learning_rate=0.001) 
            self.model.compile(optimizer=optimizer, loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            self.model.fit(self.train_generator, epochs=100, validation_data=self.test_generator, callbacks=[early_stopping])
            predictions =  self.model.predict(self.test_generator)
            self.valid_pred  = pd.Series(predictions.flatten())
            self.df_test_y = self.df_test_y[self.sequence_len:]
            self.test_y  = self.df_test_y.reset_index(drop=True)


        elif self.config['selectedType'] == "Custom":
            self.sequence_len = self.config['sequenceLength']  
            self.batch_size = self.config['batchSize']
            self.train_generator = TimeseriesGenerator(self.df_train_y.values, targets=self.df_train_y.values, length=self.sequence_len, batch_size=self.batch_size)
            self.test_generator = TimeseriesGenerator(self.df_test_y.values, targets=self.df_test_y.values, length=self.sequence_len, batch_size=self.batch_size)

            self.model = Sequential()
            for _ in range(self.config['layers'] - 1):
                self.model.add(LSTM(units=self.config['lstmNodes'], activation=self.config['activationFunction'], return_sequences=True, input_shape=(self.sequence_len, 1)))
            self.model.add(LSTM(units=self.config['lstmNodes'], activation=self.config['activationFunction']))
            self.model.add(Dense(units=50, activation=self.config['activationFunction']))
            self.model.add(Dense(units=25, activation=self.config['activationFunction']))
            self.model.add(Dense(units=1))

            optimizer = Adam(learning_rate=self.config['learningRate'])
            self.model.compile(optimizer=optimizer, loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            self.model.fit(self.train_generator, epochs=100, validation_data=self.test_generator, callbacks=[early_stopping])
            predictions =  self.model.predict(self.test_generator)
            self.valid_pred  = pd.Series(predictions.flatten())
            self.df_test_y = self.df_test_y[self.sequence_len:]
            self.test_y  = self.df_test_y.reset_index(drop=True)

        else:
            print("The type is not selected")

    def evaluate_model(self):
        mae = mean_absolute_error(self.test_y.values, self.valid_pred.values)
        mse = mean_squared_error(self.test_y.values, self.valid_pred.values)
        r2 = r2_score(self.test_y.values, self.valid_pred.values)
        mape = mean_absolute_percentage_error(self.test_y.values, self.valid_pred.values)

        self.metrics_dict = {
            'Mean Absolute Error (MAE)': mae,
            'Mean Squared Error (MSE)': mse,
            'R-squared (R^2)': r2,
            'Mean Absolute Percentage Error (MAPE)': mape
        }

        self.performance_metrics_report = pd.DataFrame.from_dict({
            'Metric': list(self.metrics_dict.keys()),
            'Value': list(self.metrics_dict.values())
        })
        self.performance_metrics_report['Value'] = self.performance_metrics_report['Value'].round(2)

    def plot_graph(self):
        self.fig = plot_actual_vs_predicted_plotly_timeseries(test_values=self.test_y, predicted_values=self.valid_pred)
    
    def log_mlflow(self):
        mlflow.set_experiment(self.config['filename'])

        with mlflow.start_run(run_name=self.key):
            mlflow.set_tag("Model", self.key)

            # for key, value in self.metrics_dict.items():
            #     mlflow.log_metric(key, value)
            # Log the LSTM model using mlflow.keras.log_model
        
            mlflow.sklearn.log_model(self.model, self.key)
            run_id = mlflow.active_run()
            
        # Save the figure and performance metrics report
        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/figure.pkl", "wb") as f:
            pickle.dump(self.fig, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/report.pkl", "wb") as f:
            pickle.dump(self.performance_metrics_report, f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/data_frame.pkl", "wb") as f:
            pickle.dump(self.df, f)
     
        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/sequence_len.pkl", "wb") as f:
            pickle.dump(self.sequence_len,f)

        with open(f"mlruns/{run_id.info.experiment_id}/{run_id.info.run_id}/target_name.pkl", "wb") as f:
            pickle.dump(self.config['yTarget'], f)
    

    def run(self):
        self.lstm_model()
        self.evaluate_model()
        self.plot_graph()
        self.log_mlflow()