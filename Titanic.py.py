# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:41:27 2024

@author: bcani
"""


import streamlit as st
import numpy as np
import pickle

# Load the model
file_path = r"C:\Users\bcani\OneDrive\Documents\Desktop\ML_deployment\logistic_model.pkl"

try:
    with open(file_path, "rb") as f:
        logistic_model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"File not found at path: {file_path}")
    st.stop()  # Stop execution if the file is not found
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Function for Titanic Survival Prediction
def Survival_prediction(input_data):
    # Prediction logic
    features = np.array([input_data])
    prediction = logistic_model.predict(features)[0]
    prediction_proba = logistic_model.predict_proba(features)[0, 1]

    return prediction, prediction_proba

# Main application
def main():
    # Title
    st.title("Titanic Survival Prediction")
    st.write("Welcome to the Titanic survival prediction app!")

    # User inputs
    st.subheader("Enter Passenger Information:")
    Pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Age = st.slider("Age", 0, 100, 25)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
    Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    Fare = st.number_input("Fare", 0.0, 500.0, 50.0)
    Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Encode categorical variables
    Sex_encoded = 1 if Sex == "Male" else 0
    Embarked_encoded = ["C", "Q", "S"].index(Embarked)

    # Create feature array
    features = [Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded]

    # Predict survival
    if st.button("Predict Survival"):
        prediction, prediction_proba = Survival_prediction(features)

        # Display results
        if prediction == 1:
            st.success(f"Prediction: Survived (Probability: {prediction_proba:.2f})")
        else:
            st.error(f"Prediction: Not Survived (Probability: {1 - prediction_proba:.2f})")


if __name__ == "__main__":
    main()
