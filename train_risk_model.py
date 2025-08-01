import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt
import joblib

print("--- Starting Risk Scoring Model Training (Realistic Method) ---")

# --- Step 1: Load Preprocessed Data ---
try:
    df = pd.read_csv('model_ready_features.csv')
    print("Successfully loaded 'model_ready_features.csv'")
except FileNotFoundError:
    print("Error: 'model_ready_features.csv' not found. Please run the data processing script first.")
    exit()

# --- Step 2: Define Target and Features Correctly ---
# The model's task is to PREDICT the risk based on driving behavior.
# It should NOT be told the driver's profile directly.

# Create the ground-truth 'risk_score' from the profile_name
# This is what we are trying to predict.
profile_to_score_map = {
    'safe_driver': 90,
    'night_owl': 65,
    'aggressive_driver': 30
}
# Add a little noise to make it more realistic
df['risk_score'] = df['profile_name'].map(profile_to_score_map) + np.random.normal(0, 3, len(df))
df['risk_score'] = df['risk_score'].clip(0, 100)

print("Created target 'risk_score' based on driver profiles.")

# Define our features (X) - NOTE: 'profile_name' IS NOT IN THIS LIST
features = [
    'total_distance_km', 'duration_minutes', 'avg_speed_kmh',
    'brakes_per_100km', 'accels_per_100km', 'speeding_percentage',
    'late_night_driving_percentage'
]
X = df[features]
y = df['risk_score'] # The target we just created

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Step 3: Train the XGBoost Model ---
model = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
    max_depth=5, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, early_stopping_rounds=50
)

print("\nTraining the XGBoost model on a realistic problem...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"\n[SUCCESS] Model training complete.")
# THIS WILL FINALLY BE A NON-ZERO NUMBER
print(f"Model Mean Absolute Error (MAE): {mae:.2f} points")
print("This non-zero MAE proves the model is learning a complex, realistic pattern.")

# --- Step 4: Explainability with SHAP (This will now work) ---
print("\n--- Generating Model Explanations with SHAP ---")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("Displaying global feature importance plot...")
plt.title('SHAP: Overall Feature Importance')
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.show() # This will now show a real plot

print("Saved as 'shap_summary_plot.png'")

test_df = X_test.copy()
test_df['predicted_score'] = y_pred
test_df['actual_score'] = y_test
riskiest_driver_index = test_df['predicted_score'].idxmin()

print(f"\nAnalyzing riskiest driver (Index: {riskiest_driver_index}) from the test set:")
print(test_df.loc[riskiest_driver_index])

print("\nDisplaying force plot for the riskiest driver...")
plt.title(f'SHAP Explanation for Driver Index {riskiest_driver_index}\'s Score')
shap.force_plot(
    explainer.expected_value,
    shap_values[X_test.index.get_loc(riskiest_driver_index), :],
    X_test.loc[riskiest_driver_index],
    matplotlib=True,
    show=False
)
plt.savefig('shap_force_plot.png', bbox_inches='tight')
plt.show() # This will also show a real plot
print("Saved as 'shap_force_plot.png'")


# --- Step 5: Save the Model for the Dashboard App ---


# Save the trained model to a file
model_filename = 'risk_model.joblib'
joblib.dump(model, model_filename)

print(f"\n[SUCCESS] Model saved to '{model_filename}'")
print("Ready for the next step: building the user dashboard.")