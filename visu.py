import pandas as pd
import streamlit as st


def load_data(file_path):
    data = pd.read_csv(file_path)  # Modify this based on your data type (e.g., CSV, Excel)
    return data
import pandas_profiling


def perform_eda(data):
    profile = pandas_profiling.ProfileReport(data)
    return profile
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
    if st.button("Perform EDA"):
        profile = perform_eda(data)
        st_profile_report(profile)

  


if __name__ == "__main__":
    main()    
