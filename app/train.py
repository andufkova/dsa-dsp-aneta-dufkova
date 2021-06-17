import app.preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np
from joblib import dump, load

MODEL_NAME = 'model_houses.joblib'

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

def fit_model(X_train, y_train):
	model = RandomForestRegressor(max_depth=6, n_estimators=140)
	model.fit(X_train, y_train)
	return model

def evaluate_model(model, X_test, y_test):
	y_pred = model.predict(X_test)
	# Replace negative predictions with 0
	y_pred = np.where(y_pred < 0, 0, y_pred)
	return compute_rmsle(y_test, y_pred)

def train_model(df, target_column, ROOT_DIR):
	return_dict = {}

	enc, enc_path, df_final, cols_con, cols_cat = app.preprocess.prepare_data(df, target_column, ROOT_DIR)

	X, y = df_final.drop(target_column, axis=1), df_final[target_column]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	#X_train.shape, X_test.shape

	model = fit_model(X_train, y_train)
	model_performance = evaluate_model(model, X_test, y_test)

	model_path = ROOT_DIR / 'models' / MODEL_NAME

	dump(model, model_path)

	return_dict['model_performance'] = model_performance
	return_dict['model_path'] = model_path
	return_dict['enc_path'] = enc_path
	return_dict['cols_con'] = cols_con
	return_dict['cols_cat'] = cols_cat

	return return_dict
