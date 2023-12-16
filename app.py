import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data_df_converted = pd.read_csv("/content/df_converted2.csv")

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

# Sliders for user input
for col in X_train.columns:
    if col != 'Weekly_Sales':  # Exclude the target variable from sliders
        min_val = float(X_train[col].min())
        max_val = float(X_train[col].max())
        default_val = float(X_train[col].mean())  # You can choose any default value here
        
        # Generate a unique key using column name and a unique suffix
        key = f"{col}_slider"
        param = st.sidebar.slider(col, key=key, min_value=min_val, max_value=max_val, value=default_val)

# User input DataFrame with selected columns
user_input = pd.DataFrame({col: [st.sidebar.slider(col, float(X_train[col].min()), float(X_train[col].max()), float(X_train[col].mean()), key=f"{col}_user_input")] for col in X_train.columns if col != 'Weekly_Sales'})

# Ensure the user_input DataFrame has the same columns as X_train
user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

# Prediction and Evaluation
# Predicting with different models based on user input
dt_prediction = dt_model.predict(user_input)
rf_prediction = rf_model.predict(user_input)
xgb_prediction = xgb_model.predict(user_input)

# Display predictions
st.write("\n**Decision Tree Prediction:**", dt_prediction[0])
st.write("\n**Random Forest Prediction:**", rf_prediction[0])
st.write("\n**XGBoost Prediction:**", xgb_prediction[0])

# Evaluate models on the test set
test_dt_preds = dt_model.predict(X_test)
rmse_dt = mean_squared_error(y_test, test_dt_preds, squared=False)
r2_dt = r2_score(y_test, test_dt_preds)

test_rf_preds = rf_model.predict(X_test)
rmse_rf = mean_squared_error(y_test, test_rf_preds, squared=False)
r2_rf = r2_score(y_test, test_rf_preds)

test_xgb_preds = xgb_model.predict(X_test)
rmse_xgb = mean_squared_error(y_test, test_xgb_preds, squared=False)
r2_xgb = r2_score(y_test, test_xgb_preds)

# Display metrics for different models on the test set
st.write("\n**Decision Tree Metrics - Test Set:**")
st.write(f"Root Mean Squared Error: {rmse_dt}")
st.write(f"R-Squared Score: {r2_dt}")

st.write("\n**Random Forest Metrics - Test Set:**")
st.write(f"Root Mean Squared Error: {rmse_rf}")
st.write(f"R-Squared Score: {r2_rf}")

st.write("\n**XGBoost Metrics - Test Set:**")
st.write(f"Root Mean Squared Error: {rmse_xgb}")
st.write(f"R-Squared Score: {r2_xgb}")

