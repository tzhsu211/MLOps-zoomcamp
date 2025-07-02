import pandas as pd
from joblib import dump, load
from evidently import DataDefinition, Dataset
from evidently.metrics import QuantileValue, DatasetMissingValueCount, DriftedColumnsCount 
from evidently.core.report import Snapshot
from evidently import Report
import logging
import psycopg
import datetime
from prefect import task, flow


logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s]: %(message)s")

with open('./models/lin_reg.bin', 'rb') as f_in:
	model = load(f_in)

target = ['duration']
num_features = ['fare_amount', 'trip_distance']
cat_features = ['PULocationID', 'DOLocationID']
time_features = ['date']

column_def = DataDefinition(
    timestamp= 'date',
    numerical_columns=num_features,
    categorical_columns= cat_features,
    datetime_columns= time_features
)

report = Report(
    metrics=[
    DatasetMissingValueCount(),
    DriftedColumnsCount(),
    QuantileValue(column = 'fare_amount', quantile = 0.5),
    QuantileValue(column = 'trip_distance', quantile = 0.5)
    ],
    include_tests= True
)

snapshot = Snapshot(
    report = report,
    name = "Snapshot",
    timestamp= 'date',
    metadata={},
    tags= []
)

create_table_statement = """
drop table if exists metrics;
create table metrics(
	timestamp timestamp,
    share_missing_values float,
	num_drifted_columns integer,
    medium_of_fare_amount float,
    medium_of_trip_distance float
)
"""

@task
def read_dataframe(filename: str)-> pd.DataFrame:
    '''
    1. read parquet file from filename
    2. calculate duration based on pickup and dropoff datetime by minutes
    3. del duration outlier (1 <= duration <= 60)
    4. change PULocationID and DOLocationID to str
    '''
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df['date'] = pd.to_datetime(df['lpep_pickup_datetime'])

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def predict(df:pd.DataFrame) -> pd.DataFrame :
    '''
    predict
    '''
    X = df[num_features + cat_features]
    y_pred = model.predict(X)
    
    df1 = X.assign(
        duration = df['duration'],
        prediction = y_pred,
        date = df['date']
    )
    
    return df1

@task
def db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=password") as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'monitoring'")
        if len(res.fetchall) ==0:
            conn.execute("create database monitoring")
        with psycopg.connect("host=localhost port=5432 user=postgres password=password") as conn:
            conn.execute(create_table_statement)
            

@task
def calculate_metrics(curr, val_df:pd.DataFrame, train_df: pd.DataFrame, date:int):
    
    '''
    curr: DB connection cursor
    val_df: validation dataframe
    train_df: training dataframe
    date: specific date (datetime.date object)
    
    1. use i to pick a date in current data
    2. change current df and refernce df to evidently dataset format and assign data definition
    3. run report and extract metric results
    4. write the results to db
    '''
    
    current_data = val_df[(val_df['date'].dt.date) == date]
    
    current_data_df = Dataset.from_pandas(current_data, data_definition=column_def)
    train_data_df = Dataset.from_pandas(train_df, data_definition=column_def)

    snapshot.run(current_data=current_data_df, reference_data= train_data_df)
    
    missing_count = snapshot['metrics'][0]['value']['count']
    drifted_count = snapshot['metrics'][1]['value']['count']
    medium_fare_amount = float(snapshot['metrics'][2]['value'])
    medium_trip_distance = float(snapshot['metrics'][3]['value'])

    curr.execute(
        """insert into metrics 
        (timestamp, share_missing_values, num_drifted_columns, medium_of_fare_amount, medium_of_trip_distance)
        VALUES (%s %s %s %s %s)""", 
        (date, missing_count, drifted_count, medium_fare_amount, medium_trip_distance)
    )
    
    logging.info(f"Metrics for {date} inserted.")
    

@flow
def batch_monitoring():
    
    
    
