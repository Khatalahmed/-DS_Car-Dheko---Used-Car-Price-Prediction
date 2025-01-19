# 🚗 Car Dekho - Used Car Price Prediction

## 📋 Project Report by Khatalahmed

## ✨ Introduction:
In this project, the task is to enhance the customer experience and optimize the pricing process for used cars by developing a machine learning model. This model, based on historical car price data, will take into account various features such as the car's make, model, year, fuel type, and transmission type. The goal is to predict the prices of used cars accurately and integrate this model into an interactive web application using Streamlit. The app will enable customers and sales representatives to input car details and receive real-time price predictions, making the car buying and selling process more efficient.

---

### 🛠️ Approach:

1. **📂 Import Files and Data Wrangling:**
    - 📥 Load datasets from multiple cities, which are in unstructured formats (e.g., text, JSON-like entries).
    - 🛠️ Use libraries like `pandas` to process each city's dataset.
    - 🧹 Parse and clean JSON-like data using `ast.literal_eval` and `pandas.json_normalize()`.
    - 🏙️ Add a column named 'City' to identify the dataset source.
    - 🔗 Merge datasets from all cities into a unified structured dataframe.
    - 💾 Save the cleaned dataframe to a CSV file.

2. **🧽 Handling Missing Values and Data Cleaning:**
    - 🗑️ Use `pandas.dropna()` to remove missing data.
    - ✂️ Remove symbols and units (₹, kmpl, CC) and clean values.
    - 🔢 Convert data to numerical formats suitable for machine learning.

3. **📊 Data Visualization:**
    - 🔍 Perform Exploratory Data Analysis (EDA) to identify patterns and relationships.
    - 🌡️ Generate correlation heatmaps to highlight significant features.
    - 🚨 Detect and handle outliers in the `price` column using the IQR method.

4. **🔎 Feature Selection:**
    - **Categorical Features:** Fuel type, body type, brand, insurance validity, color, city, and transmission.
    - **Numerical Features:** Owner number, model year, kilometers driven, mileage, engine size, and seats.

5. **⚙️ Encoding and Scaling:**
    - 🔤 Apply OneHot Encoding for categorical data.
    - 📏 Use Standard Scaler to normalize numerical features.

---

### 🤖 Model Development:

1. **✂️ Train-Test Split:**
    - Split the dataset into 75% training and 25% testing subsets.

2. **📈 Model Selection:**
    - **Linear Regression:**
        - 🧪 Baseline model for simplicity and interpretability.
        - 🛡️ Applied Ridge and Lasso regression to reduce overfitting.

    - **Gradient Boosting Regressor (GBR):**
        - 🚀 Boosts model performance by fitting new models to residuals.

    - **Decision Tree Regressor:**
        - 🌳 Handles non-linear relationships.
        - ✂️ Pruning limits tree depth to prevent overfitting.

    - **Random Forest Regressor:**
        - 🌲 Ensemble of decision trees with predictions averaged for accuracy.
        - 🎲 Features and data subsets are randomly sampled to improve generalization.

3. **📊 Model Evaluation:**
    - Models were evaluated using:
        - 📐 **Mean Squared Error (MSE):** Measures average squared errors.
        - 📏 **Mean Absolute Error (MAE):** Measures average absolute errors.
        - 📊 **R² Score:** Explains the variance proportion.

---

### 📊 Model Comparison:

| Model                   | MAE         | MSE         | RMSE        | R²          |
|-------------------------|-------------|-------------|-------------|-------------|
| LinearRegression        | 1935936589  | 5.75E+20    | 23976305799 | -4.83E+19   |
| DecisionTreeRegressor   | 1.055584958 | 2.87338617  | 1.695106536 | 0.758623737 |
| RandomForestRegressor   | 0.825905429 | 1.728796798 | 1.314837176 | 0.854773954 |
| GradientBoostingRegressor | 1.043791703 | 2.229969888 | 1.49330837  | 0.812673352 |
| RidgeRegressor          | 2.583592841 | 1.123174422 | 1.607355854 | 0.782967569 |
| LassoRegressor          | 2.585866437 | 1.12115329  | 1.608062946 | 0.782776577 |

---

### 🏆 Results:
- **🌲 Random Forest:**
    - 🥇 Best performance with the highest R² and lowest MSE/MAE.
    - 🛠️ Hyperparameter tuning optimized `n_estimators` and `max_depth`.

---

### 🛠️ Pipeline:
1. **🧩 Modular Structure:** Ensures separation of preprocessing and model training.
2. **🔧 Preprocessing:**
    - Numerical features are scaled using `StandardScaler`.
    - Categorical features are encoded with `OneHotEncoder`.
3. **⚙️ Integration:**
    - The Random Forest model is integrated, ensuring consistent preprocessing for predictions.

---

### 🌐 Model Deployment - Streamlit:
1. **Interactive Web Application:**
    - 🖱️ Intuitive interface for users to input car details (e.g., dropdowns, sliders).
    - 💬 Real-time price predictions displayed instantly using the trained model.

2. **Backend Implementation:**
    - 📦 The trained model, `StandardScaler`, and `OneHotEncoder` are loaded with `pickle`.
    - 🔄 Ensures consistent preprocessing during predictions.

---

### 🖥️ Streamlit Code for Deployment:
```python
#importing libraries
import pickle
import pandas as pd
import streamlit as slt
import numpy as np

# page setting
slt.set_page_config(layout="wide")
slt.header(':blue[CarDekho - Price Prediction 🚗]')

# Load data
df = pd.read_csv("final_model.csv")
print(df.columns)

# Streamlit interface
col1, col2 = slt.columns(2)
with col1:
    Ft = slt.selectbox("Fuel type", df['Fuel type'].unique())

    Bt = slt.selectbox("Body type", df['body type'].unique())

    Tr = slt.selectbox("Transmission", df['transmission'].unique())

    Owner = slt.selectbox("Owner", df['ownerNo'].unique())

    Brand = slt.selectbox("Brand", options=df['Brand'].unique())

    filtered_models = df[(df['Brand'] == Brand) & (df['body type'] == Bt) & (df['Fuel type'] == Ft)]['model'].unique()

    Model = slt.selectbox("Model", options=filtered_models)

    Model_year = slt.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
    
    IV = slt.selectbox("Insurance Validity", df['Insurance Validity'].unique())
    
    Km = slt.slider("Kilometers Driven", min_value=int(df['Kms Driven'].min()), max_value=int(df['Kms Driven'].max()), step=1000)

    ML = slt.number_input("Mileage", min_value=int(df['Mileage'].min()), max_value=int(df['Mileage'].max()), step=1)  

    seats = slt.selectbox("Seats", options=sorted(df['Seats'].unique()))
    
    color = slt.selectbox("Color", df['Color'].unique())

    city = slt.selectbox("City", options=df['City'].unique())

with col2:
    Submit = slt.button("Predict")

    if Submit:
        # load the model, scaler, and encoder
        with open('pipeline.pkl', 'rb') as files:
            pipeline = pickle.load(files)

        # input data
        new_df = pd.DataFrame({
            'Fuel type': Ft,
            'body type': Bt,
            'transmission': Tr,
            'ownerNo': Owner,
            'Brand': Brand,
            "model": Model,
            'modelYear': Model_year,
            'Insurance Validity': IV,
            'Kms Driven': Km,
            'Mileage': ML,
            'Seats': seats,
            'Color': color,
            'City': city
        }, index=[0])
        
        # Display the selected details
        data = [Ft, Bt, Tr, Owner, Brand, Model, Model_year, IV, Km, ML, seats, color, city]

        slt.write(data)

        # FINAL MODEL PREDICTION 
        prediction = pipeline.predict(new_df)
        slt.write(f"The price of the {new_df['Brand'].iloc[0]} car is: {round(prediction[0], 2)} lakhs")
```

---

### 🔍 Conclusion:
Deploying this predictive model via Streamlit revolutionizes the user experience by delivering swift and reliable price estimates for used cars. 🚘 It empowers customers with data-driven insights, simplifies valuation for sales representatives, and lays the foundation for future innovations like personalized recommendations and integration with real-time market data. 🌟
