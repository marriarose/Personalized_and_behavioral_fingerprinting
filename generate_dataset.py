import pandas as pd
import numpy as np

def generate_telecom_data(num_users=2000):
    """
    Generates a synthetic 'dirty' telecom usage dataset with 2000 unique IDs.
    
    Features:
    - 24-column hourly distribution (00:00 to 23:00) with raw byte counts.
    - 15% Power Users: Spikes of 4GB+ between 23:00 and 02:00.
    - 40% Office Commuters: Peaks at 09:00 and 18:00.
    - 45% Remaining Users: Random usage with 'dead zones' (0 bytes) mimicking signal drops/sleep.
    - Bursty behavior and non-uniform distributions overall.
    """
    # Set seed for reproducibility, but randomize internal variances
    np.random.seed(42)
    
    # 1. Setup IDs and 24-hour columns
    user_ids = [f"UID_{str(i).zfill(4)}" for i in range(num_users)]
    hours = [f"{str(h).zfill(2)}:00" for h in range(24)]
    
    # Calculate group sizes based on requirements
    power_user_count = int(num_users * 0.15)  # 15%
    commuter_count = int(num_users * 0.40)    # 40%
    normal_count = num_users - power_user_count - commuter_count  # Remaining 45%
    
    # Assign groups randomly
    group_labels = (
        ['power'] * power_user_count + 
        ['commuter'] * commuter_count + 
        ['normal'] * normal_count
    )
    np.random.shuffle(group_labels)
    
    # Byte conversion constants
    GB = 1024**3
    MB = 1024**2
    
    # 2. Preallocate the 24-column usage array (using int64 for large byte counts)
    data = np.zeros((num_users, 24), dtype=np.int64)
    
    # Iterate and populate based on user type
    for i in range(num_users):
        group = group_labels[i]
        
        # Base random jitter/noise for ALL users (non-uniformity)
        base_usage = np.random.randint(1*MB, 50*MB, size=24, dtype=np.int64)
        
        if group == 'power':
            # Background usage: moderately high
            usage = base_usage + np.random.randint(50*MB, 500*MB, size=24, dtype=np.int64)
            # Spikes: 23:00 to 02:00 (indices 23, 0, 1, 2)
            spike_indices = [23, 0, 1, 2]
            # Inject >4GB spikes (e.g., 4GB to 12GB)
            for idx in spike_indices:
                usage[idx] += np.random.randint(4*GB, 12*GB, dtype=np.int64)
                
        elif group == 'commuter':
            # Background usage: average
            usage = base_usage + np.random.randint(10*MB, 200*MB, size=24, dtype=np.int64)
            # Peaks at 09:00 and 18:00 (indices 9, 18)
            peak_indices = [9, 18]
            # Inject commute spikes (e.g., 500MB to 3GB)
            for idx in peak_indices:
                usage[idx] += np.random.randint(500*MB, 3*GB, dtype=np.int64)
                
        else:
            # Normal / Remaining users
            # Background usage: highly variable
            usage = base_usage + np.random.randint(0, 100*MB, size=24, dtype=np.int64)
            
            # Inject 'dead zones' (0 usage) for sleep or signal drops
            # Randomly pick 4 to 12 hours of zero usage
            num_dead = np.random.randint(4, 13)
            dead_indices = np.random.choice(24, size=num_dead, replace=False)
            for idx in dead_indices:
                usage[idx] = 0
                
        # 3. Inject random "dirty" burst anomalies for all profiles to break any remaining uniformity
        # 1-3 random hours get a sudden burst of data
        num_bursts = np.random.randint(1, 4)
        burst_hours = np.random.choice(24, size=num_bursts, replace=False)
        for bh in burst_hours:
            if usage[bh] != 0:  # Don't overwrite a dead zone if it's meant to be zero
                usage[bh] += np.random.randint(100*MB, 2*GB, dtype=np.int64)
        
        data[i] = usage
        
    # 4. Construct the DataFrame
    df = pd.DataFrame(data, columns=hours)
    df.insert(0, 'user_id', user_ids)
    
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    dataset = generate_telecom_data()
    
    # Save the dataframe to a CSV file
    output_filename = "dirty_telecom_dataset.csv"
    dataset.to_csv(output_filename, index=False)
    
    print(f"Dataset generated successfully and saved to '{output_filename}'.")
    print(f"Shape: {dataset.shape[0]} rows (IDs) x {dataset.shape[1]} columns")
    print("\nSample Preview:")
    print(dataset.head())
    
    # Basic validation printouts
    print("\n--- Validation Checks ---")
    gb_bytes = 1024**3
    print(f"Max usage in dataset (Bytes): {dataset.iloc[:, 1:].max().max():,}")
    print(f"Max usage in dataset (GB): {dataset.iloc[:, 1:].max().max() / gb_bytes:.2f} GB")
    print(f"Zero values ('dead zones') count: {(dataset.iloc[:, 1:] == 0).sum().sum():,}")
