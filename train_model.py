"""
Model Training Script for World Happiness Ranking Prediction
Trains a regression model on the World Happiness Report dataset
and serializes the trained model and StandardScaler.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Happiness.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'polls', 'HappinessModel.pickle')
SCALER_PATH = os.path.join(BASE_DIR, 'polls', 'HappinessScaler.pickle')

FEATURE_COLUMNS = [
    'Happiness Score',
    'Standard Error',
    'Economy (GDP per Capita)',
    'Family',
    'Health (Life Expectancy)',
    'Freedom',
    'Generosity',
    'Dystopia Residual'
]
TARGET_COLUMN = 'Happiness Rank'


def train_and_save():
    print(f"[*] Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

    # Validate columns
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("[*] Training GradientBoostingRegressor model...")
    model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n--- Model Performance ---")
    print(f"Train R² Score : {train_r2:.4f}")
    print(f"Test R² Score  : {test_r2:.4f}")
    print(f"Test MAE       : {test_mae:.2f}")
    print(f"Test RMSE      : {test_rmse:.2f}")

    # Test sample country prediction (Switzerland - Rank 1)
    switz_row = df[df['Country'] == 'Switzerland'][FEATURE_COLUMNS]
    switz_scaled = scaler.transform(switz_row)
    switz_pred = model.predict(switz_scaled)[0]
    print(f"\nSanity Check: Switzerland (Actual Rank: 1) -> Predicted: {switz_pred:.2f} (Clamped Rank: {max(1, min(158, int(round(switz_pred))))})")

    # Serialize scaler and model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[+] Saved StandardScaler to: {SCALER_PATH}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"[+] Saved Model to: {MODEL_PATH}")

    print("\n[SUCCESS] Model training and export completed successfully!\n")


if __name__ == '__main__':
    train_and_save()
