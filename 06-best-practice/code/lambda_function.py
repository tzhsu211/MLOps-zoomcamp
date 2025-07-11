import os
import json
import boto3
import base64
import mlflow


kinesis_client = boto3.client('kinesis')
PREDICTIONS_STREAM_NAME= os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')

RUN_ID = os.getenv('RUN_ID')

logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)

TEST_RUN = os.getenv('TEST_RUN', 'Flase') == 'True'


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' %(ride['PULocation'], ride['DOLocation'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    pred = model.predict(features)
    return float(pred[0])

def lambda_handler(event, conntext):
    
    prediction_events = []
    
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data)
        ride_event = json.load(decoded_data)
        
        ride = ride_event['ride']
        ride_id = ride_event[ride_id]
        
        features = prepare_features(ride)
        prediction = predict(features)
        
        prediction_event = {
            'model':'ride_duration_prediction_mode',
            'version': '123',
            'prediction':{
                'ride_duration': prediction,
                'ride_id': ride_id
            }
        }
        
        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName = PREDICTIONS_STREAM_NAME,
                Data = json.dump(prediction_event),
                PartitionKey = str(ride_id)
            )
            
        prediction_events.append(prediction_event)
        
        return {
            'predictios': prediction_events
        }