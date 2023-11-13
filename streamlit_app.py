import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pf
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_model
from pycaret.classification import *
from pycaret.regression import *
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, accuracy_score
import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False

###############################################################################################################

# Function to load data
# Step 1: Data Cleaning
#Data Quality: The data cleaning steps, such as handling missing values and removing duplicates,
# ensure data quality and improve the reliability of the analysis. This indicates a concern for accurate and reliable insights.

def wrangle(filepath, file_format):
    # Function to load data from different formats
    def load_data(file_path, file_format):
        if file_format == 'csv':
            return pd.read_csv(file_path)
        elif file_format == 'excel':
            return pd.read_excel(file_path)
        elif file_format == 'sql':
            # Code to load data from SQL database
            #conn = sqlite3.connect('database.db')
            pass

    # Load data using the load_data function
    data = load_data(filepath, file_format)



    duplicate_values = data.duplicated().sum()

    if duplicate_values > 0:
        # Function to handle duplicate values
        def handle_duplicate_values(df):
            # Remove duplicate rows
            df.drop_duplicates(inplace=True)
            return df

        data = handle_duplicate_values(data)

    return data
   ############################################################################################### 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def handle_Normalize_missing_values(data):
    continuous_features = []
    categorical_features = []

    # Identify the type of each feature in the data
    for column in data.columns:
        if data[column].dtype == 'object' or data[column].nunique() <= 10:
            categorical_features.append(column)
        else:
            continuous_features.append(column)

    # Fill missing values in categorical features with the mode
    for feature in categorical_features:
        data[feature].fillna(data[feature].mode()[0], inplace=True)

    # Fill missing values in continuous features with the mean
    for feature in continuous_features:
        data[feature].fillna(data[feature].mean(), inplace=True)

    # Normalize continuous features
    scaler = MinMaxScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])

    # Encode categorical features
    encoder = LabelEncoder()
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])

    return data
    ########################################################################################



def generate_histograms(df):
    for col in df.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax =sns.histplot(df[col])
        plt.title(f'Histogram of {col}')
        # ... Perform your plotting actions on the figure ...
        st.pyplot(fig)

# Function to generate box plots
def generate_box_plots(df):
    for col in df.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax =sns.boxplot(data=df[col])
        plt.title(f'Box Plot of {col}')
        # After
         
        # ... Perform your plotting actions on the figure ...
        st.pyplot(fig)

# Function to generate scatter plots
def generate_scatter_plots(df):
    numerical_cols = df.select_dtypes(include='number').columns

    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                plt.figure()
                fig, ax = plt.subplots()
                ax = sns.scatterplot(data=df, x=col1, y=col2)
                plt.title(f'Scatter Plot of {col1} vs {col2}')
                # After
                 
                # ... Perform your plotting actions on the figure ...
                st.pyplot(fig)
                
##############################################################################################################


    ############################################################################################
def main():
    st.sidebar.title('Machine Learning Package')

    # Upload data
    st.sidebar.subheader("File Selection")

    # File format selection
    file_format = st.sidebar.selectbox("Select File Format", ['csv', 'excel'])

    # File upload
    filepath = st.sidebar.file_uploader("Upload File", type=[file_format])

    if filepath is not None:
        try:
            data = wrangle(filepath, file_format)
            # Perform other preprocessing steps as needed
        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.sidebar.success('Data successfully loaded!')
        st.write(data.head())
        # Select target variable
        target_variable = st.sidebar.selectbox('Select the target variable', data.columns)

        # Split data into features and target
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]
        ###########################################################################################
        # Display a title
        st.title('Perform EDA')

        # Display EDA
        st.subheader('Exploratory Data Analysis')
        if st.button('Generate EDA'):
            if X.empty:
                st.error('The feature data is empty.')
                return
            else:
                # Generate histograms
                st.header("Histograms")
                generate_histograms(data)
                
                # Generate box plots
                st.header("Box Plots")
                generate_box_plots(data)
                
                # Generate scatter plots
                st.header("Scatter Plots")
                generate_scatter_plots(data)
            #eda_output = generate_eda(data, target_variable)
            #st.write(eda_output[0])
#############################################################################################
                # Display Statistical
               # Generate summary statistics
        st.subheader('Summary Statistics')
        if st.button('Generate Summary Statistics'):
            summary_stats = data.describe()
            st.write(summary_stats)
                    # Calculate mode
            st.subheader('Mode')
            #if st.button('Calculate Mode'):
            mode = data.mode().iloc[0]
            st.write(mode)
            ###############################################################################3


        if st.button('Train Models'):

            model_type = st.radio("Select the model type", ("Regression", "Classification"))
    
            if model_type == 'Classification':
                st.write('The case is classification')
                classification_setup(data=data, target=target_variable)
                classification_compare_models=classification_compare_models()
                st.write(classification_compare_models)
            elif model_type == 'Regression':
                print('The case is regression')
                regression_setup(data=data, target=target_variable)
                regression_compare_models=regression_compare_models()
                st.write(regression_compare_models)
     

if __name__ == '__main__':
    main()
