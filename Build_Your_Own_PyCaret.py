import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pf
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *
from pycaret.regression import *
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pycaret.datasets import get_data
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

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

def train_validate_models(X_train, y_train, X_test, y_test, model_type):
    trained_models = {}
    scores = {}

    if model_type == 'Classification':
        try:
            setup(data=X_train, target=y_train)
            exp = ClassificationExperiment()
            exp.setup(data=X_train, target=y_train)  # Replace 'target_column_name' with the actual name of your target column
            best=exp.compare_models()
            eval=evaluate_model(best)

        except Exception as e:
            print(f"An error occurred during classification model training: {str(e)}")

    elif model_type == 'Regression':
        try:
            setup(data=X_train, target=y_train)
            exp = RegressionExperiment()
            exp.setup(data=X_train, target=y_train)  # Replace 'target_column_name' with the actual name of your target column
            best=exp.compare_models()
            eval=evaluate_model(best)

        except Exception as e:
            print(f"An error occurred during regression model training: {str(e)}")

    else:
        print("Invalid model type. Please choose either 'Classification' or 'Regression'.")

    return eval, best
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
            data= handle_Normalize_missing_values(data)
            #st.dataframe(data)  # Display loaded dataset
        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.sidebar.success('Data successfully loaded!')
        st.write(data.head())
        # Select target variable
        target_variable = st.sidebar.selectbox('Select the target variable', data.columns)
        
###############################################################################################       
        # Select columns_need
        columns_need = st.sidebar.multiselect('Select the columns which you need', data.columns)
        
        if st.button('Select_columns_need'):
            data=data[columns_need]
            #st.write(data.head())
            st.write(data.columns)
        # Check if data is empty
###################################################################################
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
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if training data is empty
        if X_train.empty or y_train.empty:
            st.error('The training data is empty.')
            return

            # Select models
        models = {}

        model_type = st.radio("Select the model type", ("Regression", "Classification"))

        if model_type == 'Regression':
            train_validate_models(X_train, y_train, X_test, y_test, model_type)

        if model_type == 'Classification':
            train_validate_models(X_train, y_train, X_test, y_test, model_type)
       
if __name__ == '__main__':
    main()
