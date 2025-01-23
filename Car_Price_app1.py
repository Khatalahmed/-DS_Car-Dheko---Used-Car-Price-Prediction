# Importing necessary libraries
import pickle
import pandas as pd
import streamlit as st
import numpy as np

# Ensure sklearn is installed in your virtual environment
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
except ModuleNotFoundError:
    st.error("Required module 'sklearn' not found. Please install it using 'pip install scikit-learn'.")

# Load the pipeline using pickle
try:
    with open('pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    st.error("Pipeline file 'pipeline.pkl' not found. Ensure the file is available in the specified directory.")

# Page settings for Streamlit
st.set_page_config(page_title="CarDekho - Price Prediction", layout="wide")
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center; 
            color: #0078FF;
            font-size: 3rem;
        }
        .sub-header {
            text-align: center; 
            color: #000;
            font-size: 1.5rem;
            margin-top: -10px;
        }
        .predict-button {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #0078FF; 
            color: white; 
            font-size: 1.2rem;
            font-weight: bold; 
            text-align: center;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            cursor: pointer;
            margin: 20px auto;
        }
        .predict-button:hover {
            background-color: #0056b3;
        }
        .table-style {
            background-color: #f0f8ff;
            border: 1px solid black;
            text-align: center;
        }
        .results-section {
            font-size: 1.2rem;
            color: #0056b3;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Display headers
st.markdown("<h1 class='main-header'>🚗 CarDekho - Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Get accurate price predictions based on your car details</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Load the dataset
try:
    df = pd.read_csv(r"D:\Car_Dekho\Car_Dekho\final_model.csv")
except FileNotFoundError:
    st.error("Dataset file 'final_model.csv' not found. Ensure the file is available in the specified directory.")
else:
    # Main layout
    with st.container():
        st.markdown("<h3 style='text-align: center;'>Enter Car Details Below</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            Ft = st.selectbox("Fuel Type", df['Fuel type'].unique())
            Bt = st.selectbox("Body Type", df['body type'].unique())
            Tr = st.selectbox("Transmission", df['transmission'].unique())
            Owner = st.selectbox("Owner", df['ownerNo'].unique())
            Brand = st.selectbox("Brand", options=df['Brand'].unique())

            # Filter models dynamically
            filtered_models = df[
                (df['Brand'] == Brand) &
                (df['body type'] == Bt) &
                (df['Fuel type'] == Ft)
            ]['model'].unique()
            Model = st.selectbox("Model", options=filtered_models)

        with col2:
            Model_year = st.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
            IV = st.selectbox("Insurance Validity", df['Insurance Validity'].unique())
            Km = st.slider(
                "Kilometers Driven", 
                min_value=101, 
                max_value=550000, 
                step=5000, 
                help="Drag to select kilometers driven by the car.",
                value=275050  # Set default to mid-range
            )
            ML = st.number_input("Mileage", 
                                 min_value=int(df['Mileage'].min()), 
                                 max_value=int(df['Mileage'].max()), 
                                 step=1)
            seats = st.selectbox("Seats", options=sorted(df['Seats'].unique()))
            color = st.selectbox("Color", df['Color'].unique())
            city = st.selectbox("City", df['City'].unique())

        # Prediction section
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.markdown('<button class="predict-button">Predict Price 🚗</button>', unsafe_allow_html=True):
            # Create DataFrame for the input data
            new_df = pd.DataFrame({
                'Fuel type': [Ft],
                'body type': [Bt],
                'transmission': [Tr],
                'ownerNo': [Owner],
                'Brand': [Brand],
                'model': [Model],
                'modelYear': [Model_year],
                'Insurance Validity': [IV],
                'Kms Driven': [Km],
                'Mileage': [ML],
                'Seats': [seats],
                'Color': [color],
                'City': [city]
            })

            st.markdown("<h3 style='text-align: center; color: green;'>Prediction Results</h3>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

            # Display input details
            st.markdown("<h4>Selected Input Details:</h4>", unsafe_allow_html=True)
            st.table(new_df.style.set_properties(**{'border': '1px solid black', 'text-align': 'center'}))

            # Predict using the pipeline
            try:
                prediction = pipeline.predict(new_df)
                price_in_lakhs = round(prediction[0] / 100, 2)  # Convert price to lakhs with 2 decimal points
                st.markdown(
                    f"<div class='results-section'>Estimated Price: <strong>₹{price_in_lakhs} Lakhs</strong></div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
