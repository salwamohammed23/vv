import pandas as pd
import streamlit as st
from my_package.data_loader import load_data
from my_package.eda import perform_eda
from my_package.model_trainer import train_regression_model, train_classification_model

def load_data(file_path):
    data = pd.read_csv(file_path)  # Modify this based on your data type (e.g., CSV, Excel)
    return data
import pandas_profiling


def perform_eda(data):
    profile = pandas_profiling.ProfileReport(data)
    return profile
from pycaret.regression import *
from pycaret.classification import *


def train_regression_model(data, target_variable, models=None):
    setup(data, target=target_variable)
    best_model = compare_models(include=models)
    return best_model


def train_classification_model(data, target_variable, models=None):
    setup(data, target=target_variable)
    best_model = compare_models(include=models)
    return best_model
def main():
    st.title("My PyCaret Package Web App")

    # Data Loading
    st.header("Data Loading")
    file_path = st.file_uploader("Upload CSV file", type="csv")
    if file_path is not None:
        data = load_data(file_path)
        st.success("Data loaded successfully!")
    else:
        st.warning("Please upload a CSV file.")

    # EDA
    st.header("Exploratory Data Analysis (EDA)")
    #if st.button("Perform EDA"):
     #   profile = perform_eda(data)
      #  st_profile_report(profile)

    # Model Training
    st.header("Model Training")
    target_variable = st.selectbox("Select the target variable", data.columns)
    model_type = st.radio("Select the model type", ("Regression", "Classification"))
    selected_models = st.multiselect("Select models", ["lr", "rf", "xgboost"])

    if model_type == "Regression":
        if st.button("Train Regression Model"):
            model = train_regression_model(data, target_variable, models=selected_models)
            st.success(f"Best regression model: {model}")
    elif model_type == "Classification":
        if st.button("Train Classification Model"):
            model = train_classification_model(data, target_variable, models=selected_models)
            st.success(f"Best classification model: {model}")


if __name__ == "__main__":
    main()
