import app.preprocess
from joblib import load
import numpy as np

def replace_negative_predictions(predictions):
	predictions = np.where(predictions < 0, 0, predictions)
	return predictions

def predict(data, ROOT_DIR):
	model = load(ROOT_DIR / 'models' / 'model_houses.joblib') 
	predictions = model.predict(data)
	predictions_r = replace_negative_predictions(predictions)
	return predictions_r

def make_prediction(df, cols_con, cols_cat, ROOT_DIR):
	data = app.preprocess.prepare_data_inference(df, ROOT_DIR, cols_con, cols_cat)
	predictions = predict(data, ROOT_DIR)
	return predictions, data
