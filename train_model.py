"""
train_model.py
---------------
Trains a Linear Regression model on historical student data to predict
Final Marks from Quiz, Assignment, and Midterm scores.

Pipeline:
 1. Load dataset (students.csv)
 2. Clean data (missing values, invalid ranges, negatives, duplicates)
 3. Select features (X) and target (y)
 4. Train/test split (80/20)
 5. Train Linear Regression model
 6. Evaluate with MAE and R^2
 7. Save trained model to trained_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "students.csv"
MODEL_PATH = "trained_model.pkl"
MAX_COMPONENT_SCORE = 20  # Quiz / Assignment / Midterm max


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Drop rows with missing values
    before = len(df)
    df = df.dropna()

    # 2. Remove invalid ranges for Quiz/Assignment/Midterm (0-20) and negatives
    for col in ["Quiz", "Assignment", "Midterm"]:
        df = df[(df[col] >= 0) & (df[col] <= MAX_COMPONENT_SCORE)]

    # 3. Remove invalid Final marks (0-100)
    df = df[(df["Final"] >= 0) & (df["Final"] <= 100)]

    # 4. Drop duplicate records
    df = df.drop_duplicates()

    after = len(df)
    print(f"Data cleaning: {before} -> {after} records kept "
          f"({before - after} removed as invalid/duplicate/missing).")

    return df.reset_index(drop=True)


def main():
    df = load_and_clean_data(DATA_PATH)

    # Features and target
    X = df[["Quiz", "Assignment", "Midterm"]]
    y = df["Final"]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Training Complete")
    print(f"  Coefficients -> Quiz: {model.coef_[0]:.3f}, "
          f"Assignment: {model.coef_[1]:.3f}, Midterm: {model.coef_[2]:.3f}")
    print(f"  Intercept: {model.intercept_:.3f}")
    print(f"  MAE  (Test Set): {mae:.2f}")
    print(f"  R^2  (Test Set): {r2:.3f}")

    # Save model along with metrics so the app can display real evaluation numbers
    joblib.dump({"model": model, "mae": mae, "r2": r2}, MODEL_PATH)
    print(f"\nSaved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
