
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify


dct = {
    "vesseldwt": 500,
    "n_stevs": 5,
    "process_time": 6,
    "vesseltype_1.0": 0,

    "vesseltype_2.0": 0,
    "vesseltype_3.0": 1,
    "vesseltype_4.0": 0,
    "vesseltype_5.0": 0,

    "traveltype_ARRIVAL": 1,
    "traveltype_SHIFT": 0,
    "bulk_liquid": 1,
    "bulk_solid": 0
}

def load_model():
    model = xgb.XGBRegressor()
    model.load_model('./bestmodel_2.json')
    return model

def compose_vec(dct):
    df= pd.DataFrame([dct])
    return df

app = Flask('score')
@app.route('/score', methods=['POST'])
def predict():
    dct= request.get_json() # catches the incoming json request and converts it to a python dict
    vec= compose_vec(dct)
    predictor = load_model()
    y_pred = predictor.predict(vec)
    result= {"dicharge": float(y_pred[0])}
    return jsonify(result)

if __name__=='__main__':
    app.run(host= 'localhost', port= 9696, debug= True)
    