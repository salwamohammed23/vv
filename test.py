import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False
import seaborn as sns
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
    
def generate_histograms(df):
    for col in df.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        sns.histplot(df[col])
        plt.title(f'Histogram of {col}')
        # ... Perform your plotting actions on the figure ...
        st.pyplot(fig)

# Function to generate box plots
def generate_box_plots(df):
    for col in df.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        sns.boxplot(data=df[col])
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
                sns.scatterplot(data=df, x=col1, y=col2)
                plt.title(f'Scatter Plot of {col1} vs {col2}')
                # After
                 
                # ... Perform your plotting actions on the figure ...
                st.pyplot(fig)


def train_validate_models(X_train, y_train, X_test, y_test, model_type, selected_models):
    trained_models = {}
    scores = {}

    if model_type == 'Classification':
        try:
            clf = setup(data=X_train, target=y_train, silent=True)
            for model_name in selected_models:
                model = create_model(model_name)
                trained_model = tune_model(model)
                y_pred = predict_model(trained_model, data=X_test)
                score = accuracy_score(y_test, y_pred)
                scores[model_name] = score
                trained_models[model_name] = trained_model

        except Exception as e:
            print(f"An error occurred during classification model training: {str(e)}")

    else:
        try:
            reg = setup(data=X_train, target=y_train, silent=True)
            for model_name in selected_models:
                model = create_model(model_name)
                trained_model = tune_model(model)
                y_pred = predict_model(trained_model, data=X_test)
                score = mean_squared_error(y_test, y_pred)
                scores[model_name] = score
                trained_models[model_name] = trained_model

        except Exception as e:
            print(f"An error occurred during regression model training: {str(e)}")

    return trained_models, scores
def main():
    
    st.sidebar.title('Machine Learning Package')

    # Upload data
    st.sidebar.subheader('Data Loading')
    file = st.sidebar.file_uploader('Upload CSV', type='csv')
   
 
    if file is not None:
        data = wrangle(file)  # Assuming you have a function named `wrangle` that processes the uploaded file
                
        st.sidebar.success('Data successfully loaded!')
        st.write(data.head())
        # Select target variable
        target_variable = st.sidebar.selectbox('Select the target variable', data.columns)
        
       
        # Select drop columns
        columns_to_drop = st.sidebar.multiselect('Select the columns to drop', data.columns)
        
        if st.button('drop_columns'):
            data=data.drop(columns=columns_to_drop, axis=1, inplace=True)
            #st.write(data.head())
            st.write(data.columns)
        # Check if data is empty

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
            selected_models =  st.multiselect("Select models",["Extra Trees Regressor", 'Extreme Gradient Boosting', 'Random Forest Regressor', 'Light Gradient Boosting Machine', 'Gradient Boosting Regressor', 'Decision Tree Regressor', 'Ridge Regression', 'Lasso Regression', 'Lasso Least Angle Regression', 
                                                                  'Bayesian Ridge', 'Linear Regression', 'Huber Regressor', 'Passive Aggressive Regressor', 'Orthogonal Matching Pursuit', 'AdaBoost Regressor', '	K Neighbors Regressor', 'Elastic Net', 'Dummy Regressor', 'Least Angle Regression'])
            models.update({model: True for model in selected_models})

        if model_type == 'Classification':
            selected_models = st.multiselect("Select models",["Logistic Regression", 'K Neighbors Classifier', 'Naive Bayes', 'Decision Tree Classifier', 'SVM - Linear Kernel', 'SVM - Radial Kernel', 'Gaussian Process Classifier', 'MLP Classifier', 'Ridge Classifier', 'Random Forest Classifier', 'Ada Boost Classifier', 'Extra Trees Classifier', '	Light Gradient Boosting Machine',	'Decision Tree Classifier', 	'SVM - Linear Kernel', 'Ridge Classifier', 	'Dummy Classifier'])
            models.update({model: True for model in selected_models})

        if st.button('Train Models'):
            trained_models, scores = train_validate_models(X_train, y_train, X_test, y_test, model_type, selected_models)
        
            # Evaluate models
            st.subheader('Model Evaluation')
        
            for model_name, score in scores.items():
                st.write(f'{model_name}: {score}')
if __name__ == '__main__':
    main()
