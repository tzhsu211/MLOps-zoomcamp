import json
import requests
from deepdiff import DeepDiff

with open('./event.json', 'rt', encoding = 'utf-8') as f_in:
    event = json.load(f_in)
    
# default lambda name : data/fucntions/function(this one can be renamed)/invocations
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


print("Acutal response:")
# print(json.dumps(actual_response, indent=2))
print(actual_response)

diff = DeepDiff(actual_response, expected_response, significant_digits = 1)
print("diff: \n", diff)

assert 'type_changes' not in diff
assert 'values_changed' not in diff
