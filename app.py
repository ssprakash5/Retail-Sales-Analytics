import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Function to load data
@st.cache_data
def load_data():
    data_df_converted = pd.read_csv("/content/drive/MyDrive/df_converted2.csv")
    return data_df_converted

# Streamlit App
def main():
    st.title("Weekly Sales Prediction App")
    st.sidebar.header("Input Parameters")

    # Load data
    data_selected = load_data()

    selected_columns = ['Store', 'Type', 'Size', 'Dept', 'IsHoliday',
                        'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
                        'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Weekly_Sales']

    data_selected = data_selected[selected_columns]

    X = data_selected.drop(columns=['Weekly_Sales'])
    y = data_selected['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    user_input = {}
    for col in X_train.columns:
        if col in ['Store', 'Type', 'Dept', 'IsHoliday']:
            unique_values = X_train[col].unique()
            user_input[col] = st.sidebar.selectbox(col, unique_values)
        else:
            default_val = float(X_train[col].mean())
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=default_val)

    selected_model = st.sidebar.selectbox("Select Model", ["Decision Tree", "Random Forest", "XGBoost"])

    if selected_model == "Decision Tree":
        model = DecisionTreeRegressor()
    elif selected_model == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = XGBRegressor()

    if st.sidebar.button("Predict"):
        model.fit(X_train, y_train)
        prediction = model.predict(pd.DataFrame([user_input]))
        st.write("\n**Prediction:**", prediction[0])

if __name__ == "__main__":
    main()

