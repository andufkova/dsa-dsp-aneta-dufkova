import base64
import pandas as pd
import streamlit as st
import requests
import json

SERVER = "http://127.0.0.1:8000"

st.set_page_config(
    page_title='Diabetes prediction',
    page_icon='🍔',
    initial_sidebar_state='expanded'
)

st.title('Diabetes prediction')


my_expander = st.beta_expander('Single patient')
my_expander.header('Predict single patient', anchor='single')
col1, col2 = my_expander.beta_columns(2)

with my_expander.form(key='my_form'):
    age = col1.number_input('Age', min_value=1.0, max_value=120.0, value=48.0, format='%.1f', step=1.0)
    sex = col2.radio('Sex', ('1', '2'))
    bmi = col1.number_input('BMI', min_value=15.0, max_value=40.0, value=21.6, format='%.1f', step=0.1)
    bp = col2.number_input('Average blood presure', min_value=1.0, max_value=200.0, value=87.0, format='%f', step=1.0)
    tc = col1.number_input('T-Cells', min_value=1.0, max_value=200.0, value=183.0, format='%.1f', step=1.0)
    ldl = col2.number_input('Low-density lipoproteins', min_value=1.0, max_value=200.0, value=103.2, format='%.1f', step=0.1)
    hdl = col1.number_input('High-density lipoproteins', min_value=1.0, max_value=200.0, value=70.0, format='%f', step=1.0)
    tsh = col2.number_input('Thyroid stimulating hormone', min_value=1.0, max_value=200.0, value=3.0, format='%f', step=1.0)
    l = col1.number_input('Lamotrigine', min_value=1.0, max_value=200.0, value=3.8918, format='%.1f', step=0.1)
    bsl = col2.number_input('Blood sugar level', min_value=1.0, max_value=200.0, value=69.0, format='%f', step=0.1)

    if st.form_submit_button(label='Predict single patient'):
        request_url = f'{SERVER}/predict'
        my_json = {'age': age, 'sex': sex, 'bmi': bmi, 'bp': bp, 's1': tc, 's2': ldl, 's3': hdl, 's4': tsh, 's5': l, 's6': bsl}
        predictions = requests.post(request_url, json=my_json)
        my_expander.write('Result:')
        my_expander.write(predictions.text)
        

my_expander2 = st.beta_expander('Multiple patients')

#col1.balloons()
my_expander2.header('Predict multiple patients', anchor='multiple')
#  Upload data for all the patients
csv_file = my_expander2.file_uploader('Choose a CSV file (app is expecting file without a target value!)')
if csv_file:
    patient_diabetes_informations_df = pd.read_csv(csv_file, sep=',')
    if patient_diabetes_informations_df.shape[1] != 10:
        my_expander2.warning('The csv file has to have 10 columns. Note that you should upload a file with features without target with separator ','.')
    else:
        if my_expander2.button('Predict diabetes progression'):
            if csv_file is not None:
                json_patients = patient_diabetes_informations_df.to_json(orient="records")
                json_patients = json_patients.lower()
                request_url = f'{SERVER}/predict_patients'
                predictions = requests.post(request_url, headers={'accept': 'application/json','Content-Type': 'application/json'}, data=json_patients)
                predictions = json.loads(json.loads(predictions.text))
                patient_diabetes_informations_df['Y'] = predictions
                my_expander2.write(patient_diabetes_informations_df)

                csv = patient_diabetes_informations_df.to_csv().encode()
                b64 = base64.b64encode(csv).decode()

                href = f'<a href="data:file/csv;base64,{b64}" download="predicted.csv">Download csv file</a>'

                my_expander2.markdown(href, unsafe_allow_html=True)

            else:
                my_expander2.warning('You need to upload a csv file before')


my_expander3 = st.beta_expander('Model metadata')

if my_expander3.button('Obatin model metadata'):
    request_url = f'{SERVER}/get_metadata'
    predictions = requests.post(request_url)
    json_metadata = json.loads(json.loads(predictions.text))
    my_expander3.write(f'model: {json_metadata["name"]}\n\ndescription: {json_metadata["description"]}'
                       f'\n\ndate: {json_metadata["date"]}\n\nMSE: {json_metadata["metrics"]["mean_squared_error"]}')
