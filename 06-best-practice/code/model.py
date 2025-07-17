import os
import json
import boto3
import base64
import mlflow

def load_model(run_id:str):
    logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    ride_event = json.loads(decoded_data)
    return ride_event


class ModelService():
    
    def __init__(self, model, model_ver, callbacks = None):
        self.model = model
        self.model_ver =model_ver
        self.callbacks = callbacks or []     
        
    def prepare_features(self, ride):
        '''
         "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66,
        },
        "ride_id": 256,
        
        output for prediction
        '''
        features = {}
        features['PU_DO'] = '%s_%s' %(ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self,features):
        pred = self.model.predict(features)
        return float(pred[0])
        
    def lambda_handler(self, event):
        '''
        {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1654161514.132
                    },
                    ...
                }
            ]
        }
        '''
        prediction_events = []
        
        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            ride_event = base64_decode(encoded_data)
            
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']
            
            features = self.prepare_features(ride)
            prediction = self.predict(features)
            
            prediction_event = {
                'model':'ride_duration_prediction_mode',
                'version': self.model_ver,
                'prediction':{
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }
            
            # if not TEST_RUN:
            #     kinesis_client.put_record(
            #         StreamName = PREDICTIONS_STREAM_NAME,
            #         Data = json.dump(prediction_event),
            #         PartitionKey = str(ride_id)
            #     )
                
            for callbacks in self.call_backs:
                callbacks(prediction_event)
                
            prediction_events.append(prediction_event)
            
            return {
                'predictions': prediction_events
            }
    
    
class KinesisCallback:
    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name
        
    def put_record(self, prediction_event):
        ride_id = prediction_event['prediction']['ride_id']
        
        self.kinesis_client.put_recode(
            StreamName = self.prediction_stream_name,
            Data = json.dumps(prediction_event),
            PartitionKey = ride_id
        )

def create_kinesis_client():
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')
    
    if endpoint_url is None:
        return boto3.client('kinesis')

    return boto3.client('kinesis', endpoint_url = endpoint_url)


def init(predictions_stream_name:str, run_id: str, test_run:bool = True):
    model = load_model(run_id)
    
    callbacks = []
    
    if not test_run:
        kinesis_client = create_kinesis_client()
        kinesis_callbacks = KinesisCallback(kinesis_client, predictions_stream_name)
        callbacks.append(kinesis_callbacks.put_record)

    model_service = ModelService(model, model_ver=run_id, callbacks=callbacks) 
    
    return model_service   
    
