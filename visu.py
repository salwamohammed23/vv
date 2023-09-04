import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)  # Modify this based on your data type (e.g., CSV, Excel)
    return data
import pandas_profiling


def perform_eda(data):
    profile = pandas_profiling.ProfileReport(data)
    return profile
