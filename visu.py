
import pandas as pd
import streamlit as st
import pandas_profiling as pf
from sklearn.impute import SimpleImputer


# Function to load data
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
        # Function to handle duplicate values
        def handle_duplicate_values(data):
            # Remove duplicate rows
            data.drop_duplicates(inplace=True)
            return data

        data = handle_duplicate_values(data)

    return data


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
                profile_report = X.profile_report()
                st.write(profile_report.to_html())


if __name__ == "__main__":
    main()
