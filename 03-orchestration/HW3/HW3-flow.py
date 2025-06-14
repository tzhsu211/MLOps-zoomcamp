import sys
sys.modules.pop("numpy", None)
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

import pandas as pd
from prefect import task, flow
import pyarrow
import os
import gc
import mlflow
from tqdm import tqdm
import mlflow.sklearn

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('HW3_flowenv')

mlflow.sklearn.autolog()

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def model_training(df:pd.DataFrame, dv:DictVectorizer = None):
    
    dic= df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer()
        x = dv.fit_transform(dic)
        
    else:
        x = dv.transform(dic)       
        
    y = df['duration'].values
    
    with mlflow.start_run():
        
        training_model = LinearRegression()
        
        with tqdm(desc="Training Model"):
            training_model.fit(x, y)

        with tqdm(desc="Generating Predictions"):
            y_pred = training_model.predict(x)
        
        rmse = mean_squared_error(y, y_pred)**0.5
        
        mlflow.log_metric('RMSE', rmse)
        
        mlflow.sklearn.log_model(training_model, artifact_path="model")
        
        print(f'RMSE: {rmse}')
    
    return training_model, dv

@flow
def train_flow(filename:str):
    
    df = read_dataframe(filename=filename)
    model, dv = model_training(df)
    
    return model, dv
    
if __name__ == "__main__":
    train_flow('./Data/yellow_tripdata_2023-03.parquet')