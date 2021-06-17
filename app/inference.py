import app.preprocess
from joblib import load
import numpy as np

def replace_negative_predictions(predictions):
	predictions = np.where(predictions < 0, 0, predictions)
	return predictions

def predict(data, train_dict):
	model = load(train_dict['model_path'])
	predictions = model.predict(data)
	predictions_r = replace_negative_predictions(predictions)
	return predictions_r

def make_prediction(df, train_dict):
	data = app.preprocess.prepare_data_inference(df, train_dict)
	predictions = predict(data, train_dict)
	return predictions
