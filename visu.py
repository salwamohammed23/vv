import pandas as pd
import streamlit as st
import pandas_profiling as pf
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *
from pycaret.regression import *
from sklearn.model_selection import train_test_split
from pycaret.datasets import get_data
from sklearn.metrics import mean_squared_error, accuracy_score



# Function to load data
# Step 1: Data Cleaning
#Data Quality: The data cleaning steps, such as handling missing values and removing duplicates,
# ensure data quality and improve the reliability of the analysis. This indicates a concern for accurate and reliable insights.
def wrangle(filepath):
    # Read CSV file
    data = pd.read_csv(filepath)
    null_sum = data.isnull().sum()

    if null_sum.sum() > 0:
        # Function to handle missing values
        def handle_missing_values(df):
            imputer = SimpleImputer(strategy='mean')  # Replace missing values with column mean
            numeric_columns = data.select_dtypes(include='number').columns
            data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
            return data

        data = handle_missing_values(data)



    duplicate_values = data.duplicated().sum()

    if duplicate_values.sum() > 0:
        # Function to handle missing values
        def handle_duplicate_values(data):
            # Remove duplicate rows
            data.drop_duplicates(inplace=True)
            return data

        data = handle_duplicate_values(data)



    return data
    
# Init setup
#s = setup(data, target=target_variable, session_id=123)

@st.cache
def generate_eda(data, target_variable):

    s = setup(data=data, target=target_variable, session_id=123)
    eda_output = eda()
    return eda_output

def main():
    st.sidebar.title('Machine Learning Package')

    # Upload data
    st.sidebar.subheader('Data Loading')
    file = st.sidebar.file_uploader('Upload CSV', type='csv')

    if file is not None:
        data = wrangle(file)
        st.sidebar.success('Data successfully loaded!')
        st.write(data.head())

        # Select target variable
        target_variable = st.sidebar.selectbox('Select the target variable', data.columns)

        # Check if data is empty
        if data.empty:
            st.error('The uploaded data is empty.')
            return

        # Split data into features and target
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]

        # Display a title
        st.title('Perform EDA')

        # Display EDA
        st.subheader('Exploratory Data Analysis')
        if st.button('Generate EDA'):
            if X.empty:
                st.error('The feature data is empty.')
                return
            else:
                profile_report = data.profile_report()
                st_profile_report(profile_report)
            #eda_output = generate_eda(data, target_variable)
            #st.write(eda_output[0])
