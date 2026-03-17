import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    df = df.copy()

    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("\nModel Accuracy (R²):", score)

    return model