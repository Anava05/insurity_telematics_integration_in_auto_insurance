import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import great_circle
import random

# --- Configuration ---
DRIVER_PROFILES = {
    'safe_driver': {'speed_factor': 0.95, 'event_prob': {'brake': 0.01, 'accel': 0.01, 'turn': 0.02}},
    'aggressive_driver': {'speed_factor': 1.20, 'event_prob': {'brake': 0.1, 'accel': 0.12, 'turn': 0.08}},
    'night_owl': {'speed_factor': 1.05, 'event_prob': {'brake': 0.04, 'accel': 0.05, 'turn': 0.05}, 'time_of_day': 'night'}
}
HARSH_BRAKE_THRESHOLD = -3.0
HARSH_ACCEL_THRESHOLD = 3.0
SHARP_TURN_THRESHOLD = 2.8

def simulate_trip(driver_id, trip_id, profile_name):
    # This function is unchanged
    profile = DRIVER_PROFILES[profile_name]
    trip_data = []
    if profile.get('time_of_day') == 'night':
        start_hour = random.choice([0, 1, 2, 22, 23])
    else:
        start_hour = random.choice([7, 8, 9, 16, 17, 18])
    current_time = datetime.now().replace(hour=start_hour, minute=random.randint(0, 59))
    lat, lon = 40.7128 + random.uniform(-0.1, 0.1), -74.0060 + random.uniform(-0.1, 0.1)
    trip_duration_seconds = random.randint(300, 2700)
    speed_limit_kmh = 50
    for i in range(trip_duration_seconds):
        accel_x = np.random.normal(0, 0.3)
        accel_y = np.random.normal(0, 0.3)
        accel_z = -9.8 + np.random.normal(0, 0.1)
        if random.random() < profile['event_prob']['brake']: accel_y = np.random.uniform(-5.0, -3.1)
        elif random.random() < profile['event_prob']['accel']: accel_y = np.random.uniform(3.1, 5.0)
        elif random.random() < profile['event_prob']['turn']: accel_x = np.random.choice([-1, 1]) * np.random.uniform(3.0, 4.5)
        speed_kmh = max(0, speed_limit_kmh * profile['speed_factor'] * random.uniform(0.8, 1.2))
        distance_per_second_km = speed_kmh / 3600
        lat += distance_per_second_km / 111.1 * random.choice([-1, 1]) * 0.5
        lon += distance_per_second_km / (111.1 * np.cos(np.radians(lat))) * random.choice([-1, 1]) * 0.5
        trip_data.append({'driver_id': driver_id, 'trip_id': trip_id, 'timestamp': current_time, 'latitude': lat, 'longitude': lon, 'speed_kmh': speed_kmh, 'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z})
        current_time += timedelta(seconds=1)
    return trip_data

print("--- Starting Data Simulation ---")
all_trips_data = []
# *** THE ONLY CHANGE IN THIS SCRIPT IS WE ARE NOW STORING THE PROFILE NAME ***
trip_profiles = {}
num_trips_to_simulate = 100 # Increased for better model training
for i in range(num_trips_to_simulate):
    driver_id = f"driver_{i % 10 + 1}"
    profile_name = random.choice(list(DRIVER_PROFILES.keys()))
    trip_id = f"{driver_id}_trip_{i+1}"
    trip_profiles[trip_id] = profile_name # Store the profile for this trip
    print(f"Simulating trip {i+1}/{num_trips_to_simulate} for {driver_id} ({profile_name})")
    all_trips_data.extend(simulate_trip(driver_id, trip_id, profile_name))

raw_driving_data_df = pd.DataFrame(all_trips_data)
raw_driving_data_df.to_csv('raw_driving_data.csv', index=False)

def preprocess_for_modeling(df):
    print("\n--- Starting Preprocessing and Feature Engineering ---")
    df.dropna(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_hard_brake'] = df['accel_y'] < HARSH_BRAKE_THRESHOLD
    df['is_hard_accel'] = df['accel_y'] > HARSH_ACCEL_THRESHOLD
    df['is_sharp_turn'] = df['accel_x'].abs() > SHARP_TURN_THRESHOLD
    speed_limit = 60
    df['is_speeding'] = df['speed_kmh'] > speed_limit
    df['is_late_night'] = (df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour < 5)
    
    grouped = df.groupby('trip_id')
    features_list = []
    for trip_id, trip_df in grouped:
        coords = list(zip(trip_df['latitude'], trip_df['longitude']))
        if len(coords) < 2: continue
        total_distance_km = sum(great_circle(coords[i], coords[i+1]).kilometers for i in range(len(coords)-1))
        duration_minutes = (trip_df['timestamp'].max() - trip_df['timestamp'].min()).total_seconds() / 60
        if duration_minutes == 0 or total_distance_km < 0.1: continue
        features_list.append({
            'trip_id': trip_id, 'driver_id': trip_df['driver_id'].iloc[0],
            # *** ADD THE GROUND TRUTH PROFILE ***
            'profile_name': trip_profiles[trip_id],
            'total_distance_km': total_distance_km, 'duration_minutes': duration_minutes,
            'avg_speed_kmh': trip_df['speed_kmh'].mean(), 'hard_brake_count': trip_df['is_hard_brake'].sum(),
            'hard_accel_count': trip_df['is_hard_accel'].sum(), 'sharp_turn_count': trip_df['is_sharp_turn'].sum(),
            'brakes_per_100km': (trip_df['is_hard_brake'].sum() / total_distance_km) * 100,
            'accels_per_100km': (trip_df['is_hard_accel'].sum() / total_distance_km) * 100,
            'speeding_percentage': trip_df['is_speeding'].mean() * 100,
            'late_night_driving_percentage': trip_df['is_late_night'].mean() * 100
        })
    return pd.DataFrame(features_list)

model_ready_df = preprocess_for_modeling(raw_driving_data_df)
model_ready_df.to_csv('model_ready_features.csv', index=False)
print("\n[SUCCESS] Model-ready features saved to 'model_ready_features.csv'")