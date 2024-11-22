# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:22:48 2024

@author: bcani
"""




import streamlit as st
import numpy as np
import pickle

# Load the logistic model
file_path = r"C:\Users\bcani\OneDrive\Documents\Desktop\ML_deployment\logistic_model.pkl"

try:
    with open(file_path, "rb") as f:
        logistic_model = pickle.load(f)
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"File not found at path: {file_path}")
    logistic_model = None
except Exception as e:
    st.error(f"An error occurred: {e}")
    logistic_model = None

# Function for Titanic survival prediction
def Survival_prediction(features):
    try:
        # Convert categorical features to numeric encoding
        Sex_encoded = 1 if features[1] == "Male" else 0
        Embarked_encoded = {"C": 0, "Q": 1, "S": 2}.get(features[6], -1)

        # Create the feature array
        feature_array = np.array([[features[0], Sex_encoded, features[2], features[3], features[4], features[5], Embarked_encoded]])

        # Make predictions
        prediction = logistic_model.predict(feature_array)[0]
        prediction_proba = logistic_model.predict_proba(feature_array)[0, 1]

        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Main Streamlit app
def main():
    # Title
    st.title("Titanic Survival Prediction")
    st.write("Welcome to the survival prediction app!")

    # Input features
    Pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Age = st.slider("Age", 0, 100, 25)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
    Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    Fare = st.number_input("Fare", 0.0, 500.0, 50.0)
    Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Button for prediction
    if st.button("Predict Survival"):
        if logistic_model:
            # Call the prediction function
            prediction, prediction_proba = Survival_prediction([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked])
            
            # Display results
            if prediction == 1:
                st.success(f"Prediction: Survived (Probability: {prediction_proba:.2f})")
            else:
                st.warning(f"Prediction: Did Not Survive (Probability: {1 - prediction_proba:.2f})")
        else:
            st.error("The model could not be loaded. Please check the file path.")

# Run the app
if __name__ == "__main__":
    main()
