import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("HW2")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    with mlflow.start_run():

        mlflow.set_tag("HW2", "homework2")
        mlflow.log_param("train data", 'green_2023-01')
        mlflow.log_param("validation data", 'green_2023-02')
        mlflow.log_param("test data", 'green_2023-03')
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        max_depth=10
        random_state=0
        rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        
        # mlflow.sklearn.log_model(rf, "random_forest_model")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        rmse = root_mean_squared_error(y_val, y_pred)
        
        mlflow.log_metric("RMSE", rmse)


if __name__ == '__main__':
    run_train()
