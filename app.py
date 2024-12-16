import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained models
model = tf.keras.models.load_model('model.h5')
model2 = tf.keras.models.load_model('model2.h5')

# Load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('scaler2.pkl', 'rb') as file:
    scaler2 = pickle.load(file)

# App title
st.title('ANN application')

# --- Churn Prediction (Classification) ---
st.header('Customer Churn Prediction (Classification)')

# Widgets for user input (Classification)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geo_class')
gender = st.selectbox('Gender', label_encoder_gender.classes_, key='gender_class')
age = st.slider('Age', 18, 92, key='age_class')
balance = st.number_input('Balance', key='balance_class')
credit_score = st.number_input('Credit Score', key='credit_class')
estimated_salary = st.number_input('Estimated Salary', key='salary_class')
tenure = st.slider('Tenure', 0, 10, key='tenure_class')
num_of_products = st.slider('Number of Products', 1, 4, key='products_class')
has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='card_class')
is_active_member = st.selectbox('Is Active Member', [0, 1], key='active_class')

# Prepare input data for classification
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography' for classification
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine input features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale data for classification
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)[0][0]
st.write(f'Churn Probability: {prediction:.2f}')
if prediction > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')

# --- Salary Prediction (Regression) ---
st.header('Customer Salary Prediction (Regression)')

# Widgets for user input (Regression)
geography2 = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geo_reg')
gender2 = st.selectbox('Gender', label_encoder_gender.classes_, key='gender_reg')
age2 = st.slider('Age', 18, 92, key='age_reg')
balance2 = st.number_input('Balance', key='balance_reg')
credit_score2 = st.number_input('Credit Score', key='credit_reg')
tenure2 = st.slider('Tenure', 0, 10, key='tenure_reg')
num_of_products2 = st.slider('Number of Products', 1, 4, key='products_reg')
has_cr_card2 = st.selectbox('Has Credit Card', [0, 1], key='card_reg')
is_active_member2 = st.selectbox('Is Active Member', [0, 1], key='active_reg')
has_exited = st.selectbox("Has the customer left?", [0, 1], key='exit_reg')

# Prepare input data for regression
input_data2 = pd.DataFrame({
    'CreditScore': [credit_score2],
    'Gender': [label_encoder_gender.transform([gender2])[0]],
    'Age': [age2],
    'Tenure': [tenure2],
    'Balance': [balance2],
    'NumOfProducts': [num_of_products2],
    'HasCrCard': [has_cr_card2],
    'IsActiveMember': [is_active_member2],
    'Exited': [has_exited]
})

# One-hot encode 'Geography' for regression
geo_encoded2 = onehot_encoder_geo.transform([[geography2]]).toarray()
geo_encoded_df2 = pd.DataFrame(geo_encoded2, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine input features
input_data2 = pd.concat([input_data2.reset_index(drop=True), geo_encoded_df2], axis=1)

# Scale data for regression
input_data_scaled2 = scaler2.transform(input_data2)

# Predict salary
prediction2 = model2.predict(input_data_scaled2)
st.write(f'Estimated Salary: ${prediction2[0][0]:,.2f}')
