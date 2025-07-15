import json
import requests
from deepdiff import DeepDiff

with open('./event.json', 'rt') as f_in:
    event = json.load(f_in)
    
# default lambda name : data/fucntions/function(this one can be renamed)/invocation    
url = 'http://localhost:8080/2025-07-15/functions/function/invocations'

actual_response = requests.post(url, event)
expected_response = {
    'predictions': [
        {
            'model': 'ride_duration_prediction_model',
            'version': 'Test123',
            'prediction': {
                'ride_duration': 21.3,
                'ride_id': 256,
            },
        }
    ]
}


print("Acutal response:\n",actual_response)

diff = DeepDiff(actual_response, expected_response)
print("diff: \n", diff)

assert 'type_changes' not in diff
assert 'values_changed' not in diff
