import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model(csv_path):
    df = pd.read_csv(csv_path)

    target = "next_day_close"

    # Ensure target column is present
    if target not in df.columns:
        raise ValueError("❌ The dataset must contain a 'next_day_close' column.")

    # Keep only numeric columns but ensure target stays
    numeric_df = df.select_dtypes(include=["number"])
    if target not in numeric_df.columns:
        raise ValueError("❌ 'next_day_close' must be numeric.")

    df = numeric_df

    # Features = all numeric columns except target
    feature_cols = [col for col in df.columns if col != target]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, "trained_model.pkl")

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "features": feature_cols
    }
