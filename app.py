import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

sv_genetic = joblib.load('svc_genetic.pkl')
svc_optuna = joblib.load('svc_optuna.pkl')
rf = joblib.load('rf.pkl')
scaler = joblib.load('scaler.pkl')


st.title("Predict Titanic Survival")
st.write("Enter the details below to predict Titanic survival based on the model trained.")

Pclass = st.selectbox('Class', ['First Class', 'Second Class', 'Third Class'])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.number_input('Age', min_value=0, max_value=100, value=30)
SibSp = st.number_input('Number of siblings / spouses aboard', min_value=0, value=0)
Parch = st.number_input('Number of parents / children aboard', min_value=0, value=0)
Fare = st.number_input('Fare', min_value=0.0, value=50.0)

Pclass_map = {'First Class': 1, 'Second Class': 2, 'Third Class': 3}
Pclass = Pclass_map[Pclass]

Sex = 1 if Sex == 'male' else 0

input_data = np.array([Pclass, Sex, Age, SibSp, Parch, Fare]).reshape(1, -1)

scaled_input = scaler.transform(input_data)

prediction_1 = sv_genetic.predict(scaled_input)
prediction_2 = svc_optuna.predict(scaled_input)

if st.button('Predict'):
    if prediction_1[0] == 1:
        st.success("Prediction from SVC genetically optimized: Survived")
    else:
        st.error("Prediction from SVC genetically optimized: Not Survived")
    
    if prediction_2[0] == 1:
        st.success("Prediction from SVC optimized with Optuna: Survived")
    else:
        st.error("Prediction from SVC optimized with Optuna: Not Survived")
    
    if rf.predict(input_data)[0] == 1:
        st.success("Prediction from Random Forest: Survived")
    else:
        st.error("Prediction from Random Forest: Not Survived")