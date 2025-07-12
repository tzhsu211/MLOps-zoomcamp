import lambda_function_modify

def test_base64():
    encode = 'ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ=='
    decode = lambda_function_modify.base64_decode(encode)
    
    actual = {
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66,
        },
        "ride_id": 256,
    }
    
    assert decode == actual
    

class ModelMock:
    
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return [self.value] * len(x)
    
def test_predict():
    
    features = {
        'PU_DO': "130_205",
        'trip_distance': 3.66
    }
    
    mock_model = ModelMock(10)
    model_service = lambda_function_modify.ModelService(mock_model, '123')
    pred = model_service.predict(features)
    
    expected = 10
    
    assert pred == expected
    

def test_handler():
    
    mock_model = ModelMock(10)
    model_ver = '123'
    model_service = lambda_function_modify.ModelService(mock_model, model_ver)


    event = {
        "Records": [
            {
                "kinesis": {
                    "kinesisSchemaVersion": "1.0",
                    "partitionKey": "1",
                    "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                    "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                    "approximateArrivalTimestamp": 1654161514.132
                },
                "eventSource": "aws:kinesis",
                "eventVersion": "1.0",
                "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
                "eventName": "aws:kinesis:record",
                "invokeIdentityArn": "arn:aws:iam::387546586013:role/lambda-kinesis-role",
                "awsRegion": "eu-west-1",
                "eventSourceARN": "arn:aws:kinesis:eu-west-1:387546586013:stream/ride_events"
            }
        ]
    }
    
    actual = model_service.lambda_handler(event)
    
    expected = {
       'predictions':[{
           'model':'ride_duration_prediction_mode',
           'version': model_ver,
           'prediction':{
               'ride_duration': 10.0,
               'ride_id': 256}}
       ] 
    }
    
    assert actual == expected