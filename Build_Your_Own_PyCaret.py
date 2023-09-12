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
def load_data(file):
    data = pd.read_csv(file)
    return data
    
# Init setup
#s = setup(data, target=target_variable, session_id=123)

@st.cache
def generate_eda(data, target_variable):

    s = setup(data=data, target=target_variable, session_id=123)
    eda_output = eda()
    return eda_output


def train_models(X_train, y_train, model_type, selected_models):
    trained_models = {}

    if model_type == 'Classification':
        try:
            # Set up the classification problem
            clf = setup(data=X_train, target=y_train)
    
            for model_name in selected_models:
                model = create_model(model_name)
                #trained_model = finalize_model(model)
                #trained_models[model_name] = trained_model
    
            else:
                reg = setup(data=X_train, target=y_train)
                for model_name in selected_models:
                    model = create_model(model_name)
                    #trained_model = finalize_model(model)
                    #trained_models[model_name] = trained_model
        
            return model
        except Exception as e:
                print(f"An error occurred during classification or Reggretion model training: {str(e)}")

# Function to evaluate models
def evaluate_models(X_test, y_test, models, model_type):
    scores = {}

    for model_name, model in models.items():
        if model_type == 'Regression':
            y_pred = predict_regression_model(model, data=X_test)
            score = mean_squared_error(y_test, y_pred)
        else:
            y_pred = predict_model(model, data=X_test)
            score = accuracy_score(y_test, y_pred)

        scores[model_name] = score

    return scores

def main():
    st.sidebar.title('Machine Learning Package')

    # Upload data
    st.sidebar.subheader('Data Loading')
    file = st.sidebar.file_uploader('Upload CSV', type='csv')

    if file is not None:
        data = load_data(file)
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
            trained_models = train_models(X_train, y_train, model_type, selected_models)
        

            # Evaluate models
            scores = evaluate_models(X_test, y_test, models, model_type)
            st.subheader('Model Evaluation')

            for model_name, score in scores.items():
                st.write(f'{model_name}: {score}')
if __name__ == '__main__':
    main()
