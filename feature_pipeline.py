import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def engineer_features(df):
    """
    Engineers the 4 requested behavioral ratios from the 24-hour byte data.
    """
    # Define hour groups
    hour_cols = [f"{str(i).zfill(2)}:00" for i in range(24)]
    night_cols = ['23:00', '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00']
    day_cols = [f"{str(i).zfill(2)}:00" for i in range(7, 23)]
    prime_cols = ['19:00', '20:00', '21:00', '22:00']
    
    # 1. Night-to-Day Ratio
    night_sum = df[night_cols].sum(axis=1)
    day_sum = df[day_cols].sum(axis=1)
    night_to_day_ratio = night_sum / (day_sum + 1)  # +1 to avoid division by zero
    
    # 2. Weekend Peak Intensity
    # Since the dataset represents a 24-hour profile (no weekend column), 
    # we proxy this as the ratio of peak hour usage to average hourly usage.
    max_hour = df[hour_cols].max(axis=1)
    mean_hour = df[hour_cols].mean(axis=1)
    weekend_peak_intensity = max_hour / (mean_hour + 1)
    
    # 3. Consistency Score (standard deviation of daily use)
    consistency_score = df[hour_cols].std(axis=1)
    
    # 4. Prime-Time Concentration
    prime_sum = df[prime_cols].sum(axis=1)
    total_sum = df[hour_cols].sum(axis=1)
    prime_time_concentration = prime_sum / (total_sum + 1)
    
    # Combine into a new DataFrame
    features_df = pd.DataFrame({
        'user_id': df['user_id'],
        'Night-to-Day Ratio': night_to_day_ratio,
        'Weekend Peak Intensity': weekend_peak_intensity,
        'Consistency Score': consistency_score,
        'Prime-Time Concentration': prime_time_concentration
    })
    
    return features_df

def main():
    print("Loading raw hourly data...")
    try:
        df = pd.read_csv("dirty_telecom_dataset.csv")
    except FileNotFoundError:
        print("Error: 'dirty_telecom_dataset.csv' not found. Please ensure it exists.")
        return
        
    print("Engineering features...")
    features_df = engineer_features(df)
    
    # Columns to scale
    feature_cols = [
        'Night-to-Day Ratio', 
        'Weekend Peak Intensity', 
        'Consistency Score', 
        'Prime-Time Concentration'
    ]
    
    print("Applying RobustScaler to handle heavy-streamer outliers...")
    scaler = RobustScaler()
    # Fit and transform the feature matrix
    scaled_matrix = scaler.fit_transform(features_df[feature_cols])
    
    # Create final scaled dataframe
    scaled_df = pd.DataFrame(scaled_matrix, columns=feature_cols)
    scaled_df.insert(0, 'user_id', features_df['user_id'])
    
    # Display the Correlation Matrix
    print("\n" + "="*50)
    print("FEATURE CORRELATION MATRIX")
    print("="*50)
    corr_matrix = scaled_df[feature_cols].corr()
    print(corr_matrix.round(4).to_string())
    print("="*50)
    
    # Save the transformed output
    output_filename = "scaled_engineered_features.csv"
    scaled_df.to_csv(output_filename, index=False)
    print(f"\nPipeline complete! Scaled features saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
