import pandas as pd

from pycaret.datasets import get_data


from pycaret.regression import *
from pycaret.classification import *
# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data
def train_models(X, y, models):
    results = {}

    for model_name, model in models.items():
        model.fit(X, y)
        results[model_name] = model

    return results
# Function to test and evaluate models
def evaluate_models(X_test, y_test, models):
    scores = {}

    for model_name, model in models.items():
        if 'Regressor' in model_name:
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
        else:
            y_pred = model.predict(X_test)
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

        # Perform EDA
        perform_eda(data)

        # Split data into features and target
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_type = st.sidebar.selectbox(
            'Select classifier',
            ('Regression', 'Classification')
        )

        model_type = st.radio("Select the model type", ("Regression", "Classification"))
        selected_models = st.multiselect("Select models", ["lr", "rf", "xgboost"])

        # Select models
        models = {}
        regression_models = {
            'Random Forest Regressor': RandomForestRegressor(),
            'Linear Regression': LinearRegression()
        }
        classification_models = {
            'Random Forest Classifier': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression()
        }

        if st.checkbox('Use Regression Models'):
            models.update(regression_models)

        if st.checkbox('Use Classification Models'):
            models.update(classification_models)

        if st.button('Train Models'):
            trained_models = train_models(X_train, y_train, models)
            st.success('Models trained successfully!')

            # Evaluate models
            scores = evaluate_models(X_test, y_test, trained_models)
            st.subheader('Model Evaluation')

            for model_name, score in scores.items():
                st.write(f'{model_name}: {score}')

if __name__ == '__main__':
    main()
