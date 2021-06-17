import app.inference
import subprocess
import pandas as pd
from io import StringIO
import time

def create_submission_df(predictions, df_inference, target_column):
	df_inference[target_column] = predictions
	df_inference = df_inference[[target_column]].reset_index()
	inference_ids_df = df_inference.reset_index()[['Id']]
	submission_df = inference_ids_df.merge(df_inference, on='Id', how='left')
	return submission_df

def fill_missing_values(submission_df, target_column):
	submission_df[target_column] = submission_df[target_column].fillna(0)
	return submission_df

def save_file(filename, DATA_DIR, submission_df):
	submission_file_path = DATA_DIR / filename
	submission_df.to_csv(submission_file_path, index=False)

def send_to_kaggle(filename, DATA_DIR):
	subprocess.run('kaggle competitions submit -c house-prices-advanced-regression-techniques -f ' + str(DATA_DIR / filename) + ' -m pipeline-model', shell=True)

def get_submissions_from_kaggle():
	res = subprocess.check_output('kaggle competitions submissions -c house-prices-advanced-regression-techniques -v', shell=True)
	return res

def parse_submissions(submission):
	tmp = submission.decode('utf-8')
	data = pd.read_csv(StringIO(tmp), sep=',')
	score = float(data.head(1)['publicScore'])
	return score

def make_submission(filename, df_inference, target_column, DATA_DIR, train_dict):
	predictions = app.inference.make_prediction(df_inference, train_dict)
	submission_df = create_submission_df(predictions, df_inference, target_column)
	submission_df = fill_missing_values(submission_df, target_column)
	save_file(filename, DATA_DIR, submission_df)
	send_to_kaggle(filename, DATA_DIR)
	time.sleep(3)
	submissions = get_submissions_from_kaggle()
	score = parse_submissions(submissions)
	return score


	