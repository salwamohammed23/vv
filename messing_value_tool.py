import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Add custom CSS to hide the footer, header
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


# Function to load data
def load_data(file_path):
    _, file_extension = os.path.splitext(file_path.name)

    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError('Unsupported file format.')

# Function to handle duplicate values
def handle_duplicate_values(data):
    data.drop_duplicates(inplace=True)
    return data

# Function to handle missing values normalization
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

# Function to wrangle data
def wrangle(file_path, categorical_features_tdeal, continuous_features_tdeal):
    data = load_data(file_path)
    st.write(data.head(3))
    col1, col2 = st.columns(2)
    with col1:
        st.write('sum of nul value befor')
        st.write(data.isnull().sum())
    with col2:
        data = handle_duplicate_values(data)
        data = handle_normalize_missing_values(data, categorical_features_tdeal, continuous_features_tdeal)
        st.write('sum of nul value after')
        st.write(data.isnull().sum())
    return data
 

# Function to generate histograms
def generate_histograms(data):
    for col in data.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax =sns.histplot(data[col])
        plt.title(f'Histogram of {col}')
        # ... Perform your plotting actions on the figure ...
        st.pyplot(fig)

# Function to generate box plots
def generate_box_plots(data):
    for col in data.select_dtypes(include='number'):
        plt.figure()
        fig, ax = plt.subplots()
        ax =sns.boxplot(data=data[col])
        plt.title(f'Box Plot of {col}')
        
        st.pyplot(fig)
        
# Function to generate scatter plots
def generate_scatter_plots(df):
    numerical_cols = df.select_dtypes(include='number').columns

    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols):
            if i < j:
                plt.figure()
                fig, ax = plt.subplots()
                ax = sns.scatterplot(data=df, x=col1, y=col2, color='blue')  # يمكنك استبدال 'blue' بأي لون تريده
                plt.title(f'Scatter Plot of {col1} vs {col2}')
                # After
                 
                # ... Perform your plotting actions on the figure ...

                st.pyplot(fig)  # Move this line outside the for loop to display the final figure after all plots are generated


# Main function
def main():
    st.header("DataPrepViz")
    continuous_features_tdeal = st.sidebar.selectbox('Choose the way to treat continuous features:', ["mean()", "median()", "mode()"])
    categorical_features_tdeal = st.sidebar.selectbox("Choose the way to treat categorical features:", ["ordinal_encoder"])

    # Load data
    st.sidebar.subheader("File Selection")
    file_format = st.sidebar.selectbox("Select File Format", ['csv', 'excel'])
    file_path = st.sidebar.file_uploader("Upload File", type=[file_format])

    if file_path is not None:
        try:
            data = wrangle(file_path, categorical_features_tdeal, continuous_features_tdeal)
            st.sidebar.success('Data successfully loaded!')
            st.write('---------------------------------------------------------------------------------------------------------------------------------------')
            # Select target variable
            target_variable = st.sidebar.selectbox('Select the target variable', data.columns)

            # Split the page into two tabs
            tab1, tab2, tab3 = st.columns(3)
            
            with tab1:
                if st.button('Display visualization'):
                    generate_box_plots(data)
                    generate_scatter_plots(data)
                    generate_histograms(data)
                    
            with tab2: 
                if st.button('Download Processed Data'):
                    # Convert DataFrame to CSV and set the appropriate filename
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name='processed_data.csv',
                        mime='text/csv'
                    )
                    
            with tab3:                                            
                if st.button('Display Statistical Summary'):
                    st.write(data.describe())
                    st.write('Mode')
                    st.write(data.mode().iloc[0])

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
