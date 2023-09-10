import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from pycaret.regression import *
from sklearn.metrics import mean_squared_error, accuracy_score

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data
    
def train_models(X_train, y_train, model_type, selected_models):
    trained_models = {}

    if model_type == 'Classification':
        # Set up the classification problem
        clf = setup(data=X_train, target=y_train)
        
        for model_name in selected_models:
            model = create_model(model_name, fold=5)
            trained_model = finalize_model(model)
            trained_models[model_name] = trained_model

    else:
        reg = setup(data=X_train, target=y_train)
        for model_name in selected_models:
            model = create_model(model_name)
            trained_model = finalize_model(model)
            trained_models[model_name] = trained_model

    return trained_models

# Function to evaluate models
def evaluate_models(X_test, y_test, trained_models):
    scores = {}

    for model_name, model in trained_models.items():
        if 'Regression' in model_name:
            y_pred = predict_model(model, data=X_test)
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
        if st.sidebar.button('Generate EDA'):
            if X.empty:
                st.error('The feature data is empty.')
                return
            else:
                profile_report = data.profile_report()
                st_profile_report(profile_report)

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
                selected_models = st.multiselect("Select models", ['gbr', 'lightgbm', 'xgboost', 'lar', 'llar', 'et', 'en', 'lasso', 'knn', 
                                                                  'ada', 'omp', 'par', 'huber', 'dt', 'lr', 'br', 'rf', 'ridge', 'dummy'])
                models.update({model: True for model in selected_models})

            if model_type == 'Classification':
                selected_models = st.multiselect("Select models", ['gbc', 'lightgbm', 'ada', 'xgboost', 'lda', 'et', 'gbc', 'ada', 'knn', 'lr', 'rf', 'ridge', 'dummy'])
                models.update({model: True for model in selected_models})

            if st.sidebar.button('Train Models'):
                trained_models =train_models(X_train, y_train, model_type, selected_models)
                st.success('Models trained successfully!')

                # Evaluate models
                scores = evaluate_models(X_test, y_test, trained_models)
                st.subheader('Model Evaluation')

                for model_name, score in scores.items():
                    st.write(f'{model_name}: {score}')

if __name__ == '__main__':
    main()
