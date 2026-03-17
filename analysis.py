import pandas as pd
def load_data():
    df = pd.read_csv("data/diamonds.csv")
    return df
def basic_analysis(df):
    print("First 5 rows:\n", df.head())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSummary:\n", df.describe())