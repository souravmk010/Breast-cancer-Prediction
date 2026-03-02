# Breast-cancer-Prediction


This project is a lightweight Machine Learning web application that predicts whether a tumor is benign or malignant using a Logistic Regression model. The model is trained on selected features from the Breast Cancer dataset available in Scikit-learn and deployed using Streamlit for an interactive user interface.

The objective of this project is to demonstrate end-to-end model development, preprocessing, feature selection, model serialization, and web deployment in a structured and production-ready manner.

# Project Overview

This application includes:

Data loading using Scikit-learn's built-in Breast Cancer dataset

Feature selection to reduce input dimensionality

Data preprocessing using StandardScaler

Model training using Logistic Regression

Model serialization using Joblib

Interactive deployment using Streamlit

Real-time prediction with probability confidence score

The model is intentionally trained using only five key features to maintain simplicity, usability, and efficient performance.

# Features Used

The model is trained on the following tumor measurements:

Mean Radius

Mean Texture

Mean Perimeter

Mean Area

Mean Smoothness

# Model Details

Algorithm: Logistic Regression

Dataset: Breast Cancer dataset (Scikit-learn)

Preprocessing: StandardScaler

Model Persistence: Joblib

Approximate Accuracy: 90–95% (depending on train-test split)

# Files included:

model.py – Model training and saving script

app.py – Streamlit web application
