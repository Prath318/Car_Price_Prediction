
import pandas as pd
import numpy as np
import pickle
import streamlit as st


pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

df = pd.read_csv("Cleaned_Car_data.csv")   


st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Fill in the details below and get an estimated car price.")


st.sidebar.header("Input Car Details")

companies = sorted(df['company'].unique())
car_models = sorted(df['name'].unique())


company = st.sidebar.selectbox("Company", companies)
name = st.sidebar.selectbox("Car Name", car_models)
year = st.sidebar.number_input("Year", min_value=1990, max_value=2025, step=1, value=2019)
kms_driven = st.sidebar.number_input("KMs Driven", min_value=0, step=100, value=10000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])

if st.sidebar.button("Predict Price 💰"):
    input_data = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
    predicted_price = pipe.predict(input_data)[0]

    st.success(f"### ✅ Estimated Price: ₹ {int(predicted_price):,}")
