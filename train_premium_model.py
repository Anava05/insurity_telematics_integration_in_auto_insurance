import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import joblib

print("--- Step 2: Training Premium Optimization Model with Optuna ---")

# Load the complex dataset we just created
try:
    df = pd.read_csv('premium_features.csv')
except FileNotFoundError:
    print("Error: 'premium_features.csv' not found. Run 'generate_premium_data.py' first.")
    exit()

# --- Define Features (X) and Target (y) ---
# Note: We do NOT use zip_code directly, but the risk factors derived from it.
features = [
    'behavioral_risk_score', # The output from our first model!
    'vehicle_age_years',
    'vehicle_value_usd',
    'servicing_history_score',
    'traffic_density',
    'theft_rate',
    'annual_mileage'
]
target = 'expected_annual_loss'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Hyperparameter Optimization with Optuna ---
# This is the "braggable" part. Optuna efficiently searches for the best model settings.

def objective(trial):
    """
    This function is called by Optuna for each trial.
    It trains a model with a set of hyperparameters and returns its performance.
    """
    # Define the search space for hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 0, 5), # L2 regularization
        'alpha': trial.suggest_float('alpha', 0, 5),   # L1 regularization
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse

print("\nStarting Hyperparameter Optimization with Optuna...")
# Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize') # We want to minimize the error (RMSE)
study.optimize(objective, n_trials=50) # Run 50 different trials

# --- Train Final Model with Best Parameters ---
print("\nOptimization finished.")
print(f"Best trial found: RMSE = {study.best_value:.2f}")
print("Best hyperparameters: ", study.best_params)

# Get the best hyperparameters and retrain the final model on the full training data
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# --- Save the Final Model and its Features ---
model_filename = 'premium_model.joblib'
joblib.dump(final_model, model_filename)

features_filename = 'premium_model_features.joblib'
joblib.dump(features, features_filename)

print(f"\n[SUCCESS] Final premium model saved to '{model_filename}'")
print(f"Model feature list saved to '{features_filename}'")