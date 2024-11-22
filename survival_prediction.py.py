# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:06:51 2024

@author: bcani
"""

import streamlit as st
import numpy as np
import pickle

file_path = r"C:\Users\bcani\OneDrive\Documents\Desktop\ML_deployment\logistic_model.pkl"

try:
    with open(file_path, "rb") as f:
        logistic_model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except Exception as e:
    print(f"An error occurr")
          
 # create the input function for prediction
def Titanic (Survival_pridiction):
    
     # Input features
Pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])



# Prediction
features = np.array([[Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded]])
prediction = loaded_model.predict(features)[0]
prediction_proba = model.predict_proba(features)[0, 1]

# Display the prediction
if prediction == 1:
    st.write(f"Prediction: Survived (Probability: {prediction_proba:.2f})")
else:
    st.write(f"Prediction: Not Survived (Probability: {1 - prediction_proba:.2f})")
     
def main():
 # Title
st.title("Titanic Survival Prediction")
st.write("Welcome to the survival prediction app!")
# the input data from user
PassengerId=st.text_input('number of Passengerid')
Name=st.text_input('Name of the person')
Age=st.text_input('Age of the person')
Ticket=st.number_input('Ticket number')
Survival=st.text('Survived or not Survived')
   
# code for prediction
Survival=''

 #creating a button for prediction  
if st.button('Survival Teat Result') :
    survival=Survival_pridiction([PassengerId,Name,Age,Ticket,Survival])
    
 st.success(Survival)   
 
 if _name_=='_main_':
     main()
     