import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

"""
Note: you need to run the app from the root folder otherwise the models folder will not be found
- To run the app
$ uvicorn serving.model_as_a_service.main:app --reload
$ uvicorn serving.model_as_a_service.app:app --reload
- To make a prediction from terminal
$ curl -X 'POST' 'http://127.0.0.1:8000/predict_obj' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{ "age": 0, "sex": 0, "bmi": 0, "bp": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0, "s6": 0 }'
"""

# curl -X 'POST' 'http://127.0.0.1:8000/predict?age=48&sex=1&bmi=21.6&bp=87&s1=183&s2=103.2&s3=70&s4=3&s5=3.8918&s6=69'

app = FastAPI()
# model = joblib.load("../../models/diabetes_model.joblib")
loaded_data = joblib.load("/Volumes/data/data_science_in_production/dsa-dsp-aneta-dufkova/models/diabetes_model.joblib")
scaler = joblib.load("/Volumes/data/data_science_in_production/dsa-dsp-aneta-dufkova/models/diabetes_scaler.joblib")
model = loaded_data['model']

class DiabetesInfo(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    def as_dict(self):
        return {'age': self.age, 'sex': self.sex, 'bmi': self.bmi, 'bp': self.bp, 's1': self.s1, 's2': self.s2,
                's3': self.s3, 's4': self.s4, 's5': self.s5, 's6': self.s6}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_single_patient(diabetes_info: DiabetesInfo):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level
    model_input_data = pd.DataFrame([diabetes_info.dict()])
    model_input_data = scaler.transform(model_input_data)
    progression = model.predict(model_input_data)[0]
    return progression

@app.post("/predict_patients")
async def predict_patients(diabetes_info: List[DiabetesInfo]):
    df = pd.DataFrame([x.as_dict() for x in diabetes_info])
    df = scaler.transform(df)
    progression = model.predict(df)
    return json.dumps(progression.tolist())

@app.post("/get_metadata")
async def get_metadata():
    print(loaded_data['metadata'])
    return json.dumps(loaded_data['metadata'])
