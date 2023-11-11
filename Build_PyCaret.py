import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
import pycaret
from pycaret.classification import setup, compare_models
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models
# functin to load_data
def load_data(file_path):
    if '.' in file_path:
        _, file_extension = file_path.rsplit('.', 1)
    else:
        # If no dot is found, consider the file extension as an empty string
        file_extension = ''

    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension == 'xlsx':
        return pd.read_excel(file_path)
    elif file_extension == 'sql':
        conn = sqlite3.connect('database.db')
        # Fetch data from the database, replace 'table_name' with the actual table name
        query = 'SELECT * FROM table_name;'
        return pd.read_sql_query(query, conn)
    else:
        raise ValueError('Unsupported file format.')

#############################################################################################

def handle_duplicate_values(data):
    data.drop_duplicates(inplace=True)
    return data
######################################################################
#vigulize he data
def generate_histograms(data):
    for col in data.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax = sns.histplot(data[col])
        plt.title(f'Histogram of {col}')
        st.pyplot(fig)

def generate_box_plots(data):
    for col in data.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=data[col])
        plt.title(f'Box Plot of {col}')
        st.pyplot(fig)

def generate_scatter_plots(data):
    numerical_cols = data.select_dtypes(include='number').columns

    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                plt.figure()
                fig, ax = plt.subplots()
                ax = sns.scatterplot(data=data, x=col1, y=col2)
                plt.title(f'Scatter Plot of {col1} vs {col2}')
                st.pyplot(fig)
########################################################################
#handle_Normalize_missing_values
##################################################################################3
def handle_Normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal):
    continuous_features = []
    categorical_features = []

    for column in data.columns:
        if data[column].dtype == 'object' or data[column].nunique() <= 10:
            categorical_features.append(column)
        else:
            continuous_features.append(column)

    if categorical_features_tdeal == 'ordinal_encoder':
        ordinal_encoder = OrdinalEncoder()
        for feature in categorical_features:
            data[feature] = ordinal_encoder.fit_transform(data[feature].values.reshape(-1, 1))
    elif categorical_features_tdeal == 'imputer':
        imputer = SimpleImputer(strategy='most_frequent')
        for feature in categorical_features:
            data[feature] = imputer.fit_transform(data[feature].values.reshape(-1, 1))

    if continuous_features_tdeal == 'mean()':
        for feature in continuous_features:
            data[feature].fillna(data[feature].mean(), inplace=True)
    elif continuous_features_tdeal == 'median()':
        for feature in continuous_features:
            data[feature].fillna(data[feature].median(), inplace=True)
    elif continuous_features_tdeal == 'mode()':
        for feature in continuous_features:
            data[feature].fillna(data[feature].mode()[0], inplace=True)

    scaler = MinMaxScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])

    encoder = LabelEncoder()
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])

    return data
#####################################################333
#train_validate_models
######################################################
def train_validate_models(data, target_variable):
    continuous_features = []
    categorical_features = []

    for column in data.columns:
        if data[column].dtype == 'object' or data[column].nunique() <= 10:
            categorical_features.append(column)
        else:
            continuous_features.append(column)

    if target_variable in categorical_features:
        try:
            st.write('The case is classification')
            pycaret.classification.setup(data=data, target=target_variable)
            pycaret.classification.compare_models()
        except Exception as e:
            st.error(f"An error occurred during classification model training: {str(e)}")

    elif target_variable in continuous_features:
        try:
            st.write('The case is regression')
            regression_setup(data=data, target=target_variable)
            regression_compare_models()
        except Exception as e:
            st.error(f"An error occurred during regression model training: {str(e)}")
##############################################################################333
#apply sttreamlit
################################################333333333
def main():
    st.title('Machine Learning Package')

    # Load data
    st.sidebar.subheader("File Selection")

    file_path = st.text_input("Enter the file path:")
    data = load_data(file_path)  # Assign data here

    # Handle duplicate values
    duplicate_values = data.duplicated().sum()
    if duplicate_values > 0:
        data = handle_duplicate_values(data)

    # Handle missing values and normalization
    continuous_features_tdeal = st.selectbox("Choose the way to deal with continuous features:", ['mean()', 'median()', 'mode()'])
    categorical_features_tdeal = st.selectbox("Choose the way to deal with categorical features:", ['ordinal_encoder', 'imputer'])
    data = handle_Normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal)

    # Drop columns
    User_choice = st.radio("Do you want to drop columns?", ('Yes', 'No'))
    if User_choice == 'Yes':
        columns_to_drop = st.text_input('Enter the columns you want to drop (separated by commas):')
        columns_to_drop = columns_to_drop.split(',')
        data.drop(columns=columns_to_drop, inplace=True)

    # Show selected columns
    st.write(data.columns)

    # Select target variable
    target_variable = st.text_input('Select the target variable:')

    # Exploratory Data Analysis
    exploratory_data_analysis = st.radio("Do you want to explore your data?", ('Yes', 'No'))
    if exploratory_data_analysis == 'Yes':
        # Generate histograms
        generate_histograms(data)

        # Generate box plots
        generate_box_plots(data)

        # Generate scatter plots
        generate_scatter_plots(data)

    # Display Statistical
    display_statistical = st.radio("Do you want to get statistical summary for your data?", ('Yes', 'No'))
    if display_statistical == 'Yes':
        st.write(data.describe())
        st.write('Mode')
        st.write(data.mode().iloc[0])

    # Train models
    train_models = st.radio("Do you want to train models?", ('Yes', 'No'))
    if train_models == 'Yes':
        train_validate_models(data, target_variable)

if __name__ == '__main__':
    main()
