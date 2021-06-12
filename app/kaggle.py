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

def make_submission(filename, predictions, df_inference, target_column, DATA_DIR):
	submission_df = create_submission_df(predictions, df_inference, target_column)
	submission_df = fill_missing_values(submission_df, target_column)
	save_file(filename, DATA_DIR, submission_df)


	