#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
import pandas as pd
import click

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def predict(df, model_path, year, month):    
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    print('Model and dv loaded.')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result['prediction'] = y_pred
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df_result

@click.command()
@click.option("--year", type = int, required = True, help = 'The year of data.')
@click.option("--month", type = int, required = True, help = 'The month of data.')
def main(year, month):
    input_file = f"./data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f'./data/output/output_yellow_tripdata_{year:04d}-{month:02d}.parquet' 
    model_pth = './model.bin'
    
    df = read_data(input_file)
    print(f'{year:04d}{month:02d} Data loaded from {input_file}')
    
    df_result = predict(df, model_pth, year, month)
    print(f'Prediction complete. Writing result to {output_file}...')
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index= False
    )
    print(f'Resule saved at {output_file}.')
    print(f'The mean of predicted duration is {df_result["prediction"].mean()}.')

if __name__ == '__main__':
    main()
