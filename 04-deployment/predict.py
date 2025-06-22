import mlflow
from flask import Flask, request, jsonify
import pickle
import xgboost as xgb

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = mlflow.search_runs(experiment_names=['nyc-taxi-experiment-flow'], order_by= ['metrics.RMSE ASC']).iloc[0].run_id

model_uri = f"runs:/{run_id}/model"
model = mlflow.xgboost.load_model(model_uri)
print('Model loaded.')

with open('models/preprocessor.b', 'rb') as f_in:
    dv = pickle.load(f_in)
print('Dv loaded.')


def prepare_feature(ride):
    feature = {}
    feature['PU_DO'] = '%s_%s' %(ride['PULocationID'], ride['DOLocationID'])
    feature['trip_distance'] = ride['trip_distance']
    return feature

def predict(feature):
    x = dv.transform([feature])
    dmatrix = xgb.DMatrix(x)
    y_pred = model.predict(dmatrix)
    return float(y_pred[0])

app = Flask('Duration prediction')

@app.route('/predict', methods =['POST'])
def predict_endpoint():
    ride = request.get_json()
    feature = prepare_feature(ride)
    y_pred = predict(feature)
    
    result = {
        'PU_DO': feature['PU_DO'],
        'trip_distance': feature['trip_distance'],
        'pred_duration': y_pred
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 9696, debug=True)