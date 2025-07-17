import os
import model

PREDICTION_STREAM_NAME = os.getenv('PREDICTION_STREAM_NAME', 'ride_predictions')

RUN_ID = os.getenv('RUN_ID')

TEST_RUN = os.getenv("TEST_RUN", 'False') == 'True'


model_service= model.init(
    predictions_stream_name=PREDICTION_STREAM_NAME,
    run_id= RUN_ID,
    test_run= TEST_RUN
)

def lambda_handler(event):
    return model_service.lambda_handler(event)
    