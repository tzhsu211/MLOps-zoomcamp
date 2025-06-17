import mlflow
from flask import Flask, request, jsonify


model_path = "mlruns/2/models/m-2728a9c5bc9841588e574a9a8c8fa52b/artifacts"
model = mlflow.pyfunc.load_model(model_path)
print('Model Loaded.')

def prepare_feature(ride):
    feature = {}
    feature['PU_DO'] = '%s_%s' %(ride['PULocationID'], ride['DOLocationID'])
    feature['trip_distance'] = ride['trip_distance']
    
    
    return feature

def predict(x):
    y_pred = model.predict(x)
    return y_pred

app = Flask('Duration prediction')

@app.route('/predict', methods =['POST'])
def predict_endpoint():
    ride = request.get_json()
    
    feature = prepare_feature(ride)
    y_pred = predict(feature)
    
    result = {
        'PU_DO': feature['PU_DO'],
        'trip_distance': feature['trip_distance'],
        'duration': y_pred,
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 9696, debug=True)