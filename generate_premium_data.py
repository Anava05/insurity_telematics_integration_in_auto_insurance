import pandas as pd
import numpy as np

print("--- Step 1: Generating Complex Premium Dataset ---")

# Load the telematics data to get driver IDs and their average risk scores
try:
    telematics_df = pd.read_csv('model_ready_features.csv')
except FileNotFoundError:
    print("Error: 'model_ready_features.csv' not found. Run 'process_data.py' first.")
    exit()

# We need the same driver profiles to create a logical link
profile_to_score_map = {
    'safe_driver': 90,
    'night_owl': 65,
    'aggressive_driver': 30
}
telematics_df['behavioral_risk_score'] = telematics_df['profile_name'].map(profile_to_score_map) + np.random.normal(0, 3, len(telematics_df))
telematics_df['behavioral_risk_score'] = telematics_df['behavioral_risk_score'].clip(lower=0, upper=100) # Corrected here too

# Aggregate to get one 'behavioral_risk_score' per driver
driver_risk_scores = telematics_df.groupby('driver_id')['behavioral_risk_score'].mean().reset_index()

# --- Create a new, rich dataset for premium calculation ---
num_drivers = len(driver_risk_scores)
np.random.seed(42) # Ensure reproducibility

# Feature Generation - This is where we add complexity
# 1. Vehicle Features
vehicle_age_years = np.random.randint(0, 15, size=num_drivers)
vehicle_value_usd = np.random.randint(5000, 80000, size=num_drivers)
servicing_history_score = np.random.uniform(0.5, 1.0, size=num_drivers) # 1.0 = perfect history

# 2. Location Features (Simulating external data joins)
zip_codes = [f"{90200 + i}" for i in range(num_drivers)]
# Create a "lookup table" for location-based risk
zip_risk_map = {
    zip_code: {
        'traffic_density_factor': np.random.uniform(1.0, 1.5),
        'theft_rate_factor': np.random.uniform(1.0, 2.0)
    } for zip_code in zip_codes
}

# 3. Driver Features
annual_mileage = np.random.randint(5000, 25000, size=num_drivers)

# Assemble the DataFrame
premium_df = driver_risk_scores.copy()
premium_df['vehicle_age_years'] = vehicle_age_years
premium_df['vehicle_value_usd'] = vehicle_value_usd
premium_df['servicing_history_score'] = servicing_history_score
premium_df['location_zip_code'] = zip_codes
premium_df['traffic_density'] = premium_df['location_zip_code'].apply(lambda z: zip_risk_map[z]['traffic_density_factor'])
premium_df['theft_rate'] = premium_df['location_zip_code'].apply(lambda z: zip_risk_map[z]['theft_rate_factor'])
premium_df['annual_mileage'] = annual_mileage

# --- Engineer the Target Variable: 'Expected_Annual_Loss' ---
# This formula is complex and non-linear, making it a challenging and realistic ML problem.
# It simulates what an actuary might model.

# Base loss is a small percentage of vehicle value
base_loss = premium_df['vehicle_value_usd'] * 0.02

# Risk from behavior (highly non-linear: low scores are punished exponentially)
behavioral_risk_multiplier = np.exp((100 - premium_df['behavioral_risk_score']) / 25)

# Risk from other factors
mileage_risk = premium_df['annual_mileage'] * 0.01
location_risk = (premium_df['traffic_density'] + premium_df['theft_rate']) * 50
vehicle_age_risk = premium_df['vehicle_age_years']**1.5 * 3 # Risk increases non-linearly with age
servicing_penalty = (1 - premium_df['servicing_history_score']) * 200

# Combine all factors to get the final loss
premium_df['expected_annual_loss'] = (
    base_loss + mileage_risk + location_risk + vehicle_age_risk + servicing_penalty
) * behavioral_risk_multiplier

# Add final random noise
premium_df['expected_annual_loss'] += np.random.normal(0, 50, size=num_drivers)

# --- THIS IS THE CORRECTED LINE ---
premium_df['expected_annual_loss'] = premium_df['expected_annual_loss'].clip(lower=50) # Loss can't be negative

# Save to a new file
premium_df.to_csv('premium_features.csv', index=False)

print("\n[SUCCESS] Complex premium dataset created and saved to 'premium_features.csv'")
print("Dataset Head:\n", premium_df.head())