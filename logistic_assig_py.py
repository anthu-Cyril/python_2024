# -*- coding: utf-8 -*-
"""logistic_assig.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1D4yxnVD8FvviqXztrTRqxVupBhG_o29B
"""

## LOGISTIC REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#  1. Data Exploration:

data = pd.read_csv("/content/Titanic_train.csv")

# Examine the features and their types
print("Dataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe(include="all"))

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Histograms for numerical features
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
data[numerical_features].hist(bins=15, figsize=(15, 10), edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.show()

# Box plots of numerical features against 'Survived'
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Survived', y=feature, data=data)
    plt.title(f'Boxplot of {feature} by Survival')
plt.tight_layout()
plt.show()

# Count plots for categorical features
categorical_features = ['Pclass', 'Sex', 'Embarked']
plt.figure(figsize=(15, 5))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    sns.countplot(x=feature, hue='Survived', data=data)
    plt.title(f'Countplot of {feature} by Survival')
plt.tight_layout()
plt.show()

# Pair plot for feature relationships
selected_features = ['Age', 'Fare', 'Pclass', 'Survived']
sns.pairplot(data[selected_features], hue='Survived', diag_kind='kde', corner=True)
plt.show()

# Correlation heatmap
# Select only numerical features for correlation calculation
numerical_data = data.select_dtypes(include=['number'])  # Select only numerical columns

correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 2.Data Preprocessing

from sklearn.impute import SimpleImputer

data['Age'] = SimpleImputer(strategy='median').fit_transform(data[['Age']])
# The following line has been modified to extract the 1D array from the 2D array returned by fit_transform.
data['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(data[['Embarked']])[:, 0] # Extract the first (and only) column
data = data.drop(columns=['Cabin'])  # Drop Cabin due to excessive missing values

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
for col in ['Sex', 'Embarked']:
    data[col] = LabelEncoder().fit_transform(data[col])

# 3. Model Building:

# Feature selection
X = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Save the model using pickle
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# loading the saved model
# Use 'latin1' encoding, which is a common alternative for pickled files
loaded_model = pickle.load(open('logistic_model.pkl', 'rb'), encoding='latin1')

# ---- Streamlit Deployment ----
# Save the following code in a file named "app.py"

!pip install streamlit

# Load the trained model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Titanic Survival Prediction")

import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Titanic Survival Prediction")

# Input features
Pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs
Sex_encoded = 1 if Sex == "Male" else 0
Embarked_encoded = {"C": 0, "Q": 1, "S": 2}[Embarked]

# Prediction
features = np.array([[Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded]])
prediction = loaded_model.predict(features)[0]
prediction_proba = model.predict_proba(features)[0, 1]

# Display the prediction
if prediction == 1:
    st.write(f"Prediction: Survived (Probability: {prediction_proba:.2f})")
else:
    st.write(f"Prediction: Not Survived (Probability: {1 - prediction_proba:.2f})")