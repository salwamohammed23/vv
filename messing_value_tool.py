import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
import os

# functin to load_data
def load_data(file_path):
    _, file_extension = os.path.splitext(file_path.name)

    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    #elif file_extension == 'sql':
       # conn = sqlite3.connect('database.db')
        # TODO: Implement SQL data loading logic
        pass
    else:
        raise ValueError('Unsupported file format.')

def handle_duplicate_values(data):
    # Function to handle duplicate values
    data.drop_duplicates(inplace=True)
    return data

def handle_normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal):
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

def wrangle(file_path, categorical_features_tdeal, continuous_features_tdeal):
    data = load_data(file_path)
    st.write('sum of nul value befor')
    st.write(data.isnull().sum())
    st.write('--------------------------------------------------------------------------------------------')
    data =handle_duplicate_values(data)
    data = handle_normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal)
    st.write('sum of nul value after')

    st.write(data.isnull().sum())
    return data

######################################################################
#vigulize he data
def generate_histograms(data):
    for col in data.select_dtypes(include='number'):
        fig, ax = plt.subplots()
        sns.histplot(data[col], ax=ax)
        st.pyplot(fig)
        st.title(f'Histogram of {col}')
        plt.savefig('path/to/save/figure.png')

def generate_box_plots(data):
    for col in data.select_dtypes(include='number'):
        fig, ax = plt.subplots()
        sns.boxplot(data[col], ax=ax)
        st.pyplot(fig)
        st.title(f'boxplot of {col}')
        plt.savefig('path/to/save/figure.png')


def generate_scatter_plots(data):
    numerical_cols = data.select_dtypes(include='number').columns

    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                st.pyplot(sns.scatterplot(data=data, x=col1, y=col2))
                st.title(f'Scatter Plot of {col1} vs {col2}')
########################################################################

#apply sttreamlit
################################################333333333
def main():
    continuous_features_tdeal  = st.sidebar.selectbox('choose the way to treat continuous features :',  ["mean()", "median()", "mode()"])
    categorical_features_tdeal = st.sidebar.selectbox("choose the way to treat categorical features  :",  ["ordinal_encoder"])

    

    # Load data
    st.sidebar.subheader("File Selection")

    # File format selection
    file_format = st.sidebar.selectbox("Select File Format", ['csv', 'excel'])

    # File upload
    file_path = st.sidebar.file_uploader("Upload File", type=[file_format])

    if file_path is not None:
        try:
            data = wrangle(file_path, categorical_features_tdeal, continuous_features_tdeal)
            st.sidebar.success('Data successfully loaded!')
            st.write(data.head())
                # Select target variable
            target_variable = st.sidebar.selectbox('Select the target variable', data.columns)
                # Exploratory Data Analysis
            #exploratory_data_analysis = st.radio("Do you want to explore your data?", ('Yes', 'No'))
            #if exploratory_data_analysis == 'Yes':
                    # Generate histograms
                #generate_histograms(data)

                    # Generate box plots
                #generate_box_plots(data)

                    # Generate scatter plots
                #generate_scatter_plots(data)

                # Display Statistical
            display_statistical = st.radio("Do you want to get statistical summary for your data?", ('Yes', 'No'))
            if display_statistical == 'Yes':
                st.write(data.describe())
                st.write('Mode')
                st.write(data.mode().iloc[0])
            # Perform other preprocessing steps as needed
        except Exception as e:
            st.error(f"Error: {str(e)}")


     

    #####################################################################
 
  
  

  

if __name__ == '__main__':
    main()
