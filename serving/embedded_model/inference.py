import joblib
import numpy as np
import pandas as pd


def predict(diabetes_dataframe: pd.DataFrame) -> np.ndarray:
    # https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    # for caching data: https://docs.streamlit.io/en/stable/tutorial/create_a_data_explorer_app.html
    model = joblib.load('../../models/diabetes_model.joblib')
    predictions = model.predict(diabetes_dataframe)
    return predictions

