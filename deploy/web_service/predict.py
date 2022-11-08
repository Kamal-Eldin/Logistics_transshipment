import xgboost as xgb
model = xgb.XGBRegressor()

model.load_model('./bestmodel_2.json')


print(model.get_params())