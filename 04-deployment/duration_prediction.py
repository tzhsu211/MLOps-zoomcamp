import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
import pickle
from prefect import task, flow
import mlflow.xgboost

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('nyc-taxi-experiment-flow')
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@task
def file_load(path:str) -> pd.DataFrame:
    
    cols = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PULocationID', 'DOLocationID',
       'trip_distance']
    df = pd.read_parquet(path, columns=cols)
    
    df['duration'] = (df['lpep_dropoff_datetime']-df['lpep_pickup_datetime']).dt.total_seconds()/60
    df = df[(df['duration']>=1)&(df['duration']<=60)]
    
    df['PU_DO'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)
    
    return df[['duration', 'PU_DO', 'trip_distance']]

@task
def X_feature(df:pd.DataFrame, dv:DictVectorizer = None):
    
    dic = df[['PU_DO', 'trip_distance']].to_dict(orient = 'records')
    
    if dv is None:
        dv = DictVectorizer()
        x = dv.fit_transform(dic)
        
    else:
        x = dv.transform(dic)
    
    y = df['duration']
        
    return x, y, dv


@task
def model_training(X_train, y_train, X_val, y_val, dv):
    
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label = y_train)
        val = xgb.DMatrix(X_val, label = y_val)
        
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        
        mlflow.log_params(best_params)
        
        booster = xgb.train(params=best_params, dtrain=train, num_boost_round=30, evals=[(val, 'validation')], early_stopping_rounds=20)
        
        y_pred = booster.predict(val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('RMSE', rmse)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path='model')
        
    return booster, dv


@flow
def training_flow(train_set, val_set):
    
    train_df = file_load(train_set)
    val_df = file_load(val_set)
    
    X_train, y_train, dv = X_feature(train_df)
    X_val, y_val, dv = X_feature(val_df, dv)
    
    booster, dv = model_training(X_train, y_train, X_val, y_val, dv)
    
    return booster, dv

if __name__ == '__main__':
    training_flow('../Data/green_tripdata_2021-01.parquet', '../Data/green_tripdata_2021-02.parquet')