import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the models and scaler
try:
    meta_model = joblib.load('Stacked_Model.joblib')
    logistic_regression = joblib.load('LogisticRegression.joblib')
    svm = joblib.load('SVM.joblib')
    tree = joblib.load('DecisionTree.joblib')
    forest = joblib.load('RandomForest.joblib')
    scaler = joblib.load('scaler.joblib')  # Load the scaler
    st.write("Models Loaded Sucessfully")
except FileNotFoundError:
    st.error("One or more model files not found. Please ensure they are in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Define the input parameters (Corrected and Consistent)
input_parameters = ['ph', 'hardness', 'turbidity', 'arsenic', 'chloramine', 'bacteria', 'lead', 'nitrates', 'mercury']

# Create a function to predict the water quality
def predict_water_quality(input_data):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure correct column order and names
        input_df = input_df[input_parameters]

        # Scale the input data
        X_scaled = scaler.transform(input_df)

        # Get predictions from base models (probabilities)
        log_reg_pred = logistic_regression.predict_proba(X_scaled)[:, 1]
        svm_pred = svm.predict_proba(X_scaled)[:, 1]
        tree_pred = tree.predict_proba(X_scaled)[:, 1]
        forest_pred = forest.predict_proba(X_scaled)[:, 1]

        # Create a stacked input for the meta-model
        stacked_input = np.array([log_reg_pred[0], svm_pred[0], tree_pred[0], forest_pred[0]]).reshape(1, -1)

        # Get the final prediction from the meta-model
        final_prediction = meta_model.predict(stacked_input)[0]

        return final_prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Create the Streamlit web app
st.title('Safeguarding Public Health Through Comprehensive Water Purity Analysis')

# Create input fields for each parameter
input_data = {}
for param in input_parameters:
    input_data[param] = st.number_input(f'Enter {param} value', value=0.0)

# Predict water quality when the 'Predict' button is clicked
if st.button('Predict'):
    prediction = predict_water_quality(input_data)

    if prediction is not None:
        if prediction == 0:
            st.write('The water is predicted to be: Unsafe')
        else:
            st.write('The water is predicted to be: Safe')
