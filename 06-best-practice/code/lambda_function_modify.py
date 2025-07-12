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
    


# kinesis_client = boto3.client('kinesis')
# PREDICTIONS_STREAM_NAME= os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
# TEST_RUN = os.getenv('TEST_RUN', 'Flase') == 'True'


class ModelService():
    
    
    def __init__(self, model, model_ver):
        self.model = model
        self.model_ver =model_ver
        
        
    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = '%s_%s' %(ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self,features):
        pred = self.model.predict(features)
        return float(pred[0])
        
    def lambda_handler(self, event):
    
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
                
            prediction_events.append(prediction_event)
            
            return {
                'predictions': prediction_events
            }
    
def init(predictions_stream_name:str, run_id: str, test_run:bool = True):
    model = load_model(run_id)
    model_service = ModelService(model)
    
    