import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# تحميل البيانات
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

df = load_data("/content/first_inten_project.csv")

# معالجة البيانات
@st.cache_data
def preprocess_data(df):
    df = df[['Booking_ID', 'number of adults', 'number of children',
             'number of weekend nights', 'number of week nights', 'type of meal',
             'car parking space', 'room type', 'lead time', 'market segment type',
             'repeated', 'P-C', 'P-not-C', 'average price ', 'special requests',
             'date of reservation', 'booking status']]
    
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    df.drop_duplicates(inplace=True)
    
    encoder = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = encoder.fit_transform(df[col].astype(str))
    
    return df

df = preprocess_data(df)

# تقسيم الصفحة إلى Sidebar
st.sidebar.title("🔍 استكشاف البيانات")
st.sidebar.write("## 🗂️ عرض البيانات الأساسية")
st.sidebar.dataframe(df.head())

st.sidebar.write("## 📊 التحليل الإحصائي")
st.sidebar.write(df.describe())

st.sidebar.write("## 📈 الرسوم البيانية التفاعلية")
fig = px.scatter_3d(df, x='lead time', y='number of adults', z='average price ', color='booking status')
st.sidebar.plotly_chart(fig)

st.sidebar.write("## 🔗 تحليل الارتباطات")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.sidebar.pyplot(fig)
