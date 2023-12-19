import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data_df_converted = pd.read_csv("/content/drive/MyDrive/df_converted2.csv")

# Selecting specific columns for modeling
selected_columns = ['Store', 'Type', 'Size', 'Dept', 'IsHoliday',
                    'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
                    'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Weekly_Sales']

data_selected = data_df_converted[selected_columns]

# Splitting into features and target variable
X = data_selected.drop(columns=['Weekly_Sales'])  # Features
y = data_selected['Weekly_Sales']  # Target variable

# Splitting the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Model Training
# Decision Tree Model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost Model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Streamlit App
st.title("Weekly Sales Prediction App")
st.sidebar.header("Input Parameters")

# Selectbox and Sliders for user input
user_input = {}
for col in X_train.columns:
    if col in ['Store', 'Type', 'Dept', 'IsHoliday']:
        unique_values = X_train[col].unique()
        user_input[col] = st.sidebar.selectbox(col, unique_values)
    elif col != 'Weekly_Sales':
        default_val = float(X_train[col].mean())
        min_val = float(X_train[col].min())
        max_val = float(X_train[col].max())
        user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=default_val)

# Selecting the model
selected_model = st.sidebar.selectbox("Select Model", ["Decision Tree", "Random Forest", "XGBoost"])

if selected_model == "Decision Tree":
    model = dt_model
elif selected_model == "Random Forest":
    model = rf_model
else:
    model = xgb_model

# Predict button
if st.sidebar.button("Predict"):
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame([user_input])
    user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

    # Prediction
    prediction = model.predict(user_input_df)

    # Display prediction
    st.write("\n**Prediction:**", prediction[0])
