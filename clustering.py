import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress standard sklearn warnings for clean output
warnings.filterwarnings('ignore')

def main():
    print("Loading scaled features...")
    try:
        scaled_df = pd.read_csv("scaled_engineered_features.csv")
    except FileNotFoundError:
        print("Error: 'scaled_engineered_features.csv' not found. Run the feature pipeline first.")
        return
        
    # Drop user_id for clustering
    X = scaled_df.drop('user_id', axis=1)
    
    # 1. Loop K=2 to 10
    k_values = range(2, 11)
    inertias = []
    silhouettes = []
    
    print("Running K-Means++ and evaluating K from 2 to 10...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        
    # 2. Plotting Inertia and Silhouette Coefficient in subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inertia Subplot
    axes[0].plot(k_values, inertias, marker='o', linestyle='-', color='b')
    axes[0].set_title('Elbow Method (Inertia)')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_xticks(k_values)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Silhouette Subplot
    axes[1].plot(k_values, silhouettes, marker='o', linestyle='-', color='g')
    axes[1].set_title('Silhouette Coefficient')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xticks(k_values)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_filename = 'kmeans_evaluation.png'
    plt.savefig(plot_filename)
    print(f"Saved evaluation plot to '{plot_filename}'.")
    
    # 3. Find Best K 
    # The 'natural break' or plateau where Silhouette score is maximized
    best_k = k_values[np.argmax(silhouettes)]
    print(f"\nOptimal K identified (Highest Silhouette Score): {best_k}")
    
    # 4. Final K-Means fit with optimal K
    print(f"Fitting final K-Means model with K={best_k}...")
    final_kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
    final_labels = final_kmeans.fit_predict(X)
    
    # 5. Append cluster labels to the ORIGINAL dataframe
    print("Loading original raw byte dataset...")
    orig_df = pd.read_csv("dirty_telecom_dataset.csv")
    orig_df['Cluster'] = final_labels
    
    output_filename = "clustered_telecom_dataset.csv"
    orig_df.to_csv(output_filename, index=False)
    print(f"Appended labels and saved back to '{output_filename}'.")
    
    # 6. Print summary of the 'Centroid' - Average Hourly Peak
    print("\n" + "="*50)
    print("CLUSTER CENTROID SUMMARY: AVERAGE HOURLY PEAK")
    print("="*50)
    
    hour_cols = [f"{str(i).zfill(2)}:00" for i in range(24)]
    GB = 1024**3
    MB = 1024**2
    
    # Iterate through each cluster to find their peak usage times
    for cluster_id in range(best_k):
        cluster_data = orig_df[orig_df['Cluster'] == cluster_id]
        
        # Calculate the mean byte usage for each hour across all users in this cluster
        avg_hourly_usage = cluster_data[hour_cols].mean()
        
        # Find the peak hour and its volume
        peak_hour = avg_hourly_usage.idxmax()
        peak_volume = avg_hourly_usage.max()
        
        # Human-readable formatting
        if peak_volume >= GB:
            vol_str = f"{peak_volume / GB:.2f} GB"
        else:
            vol_str = f"{peak_volume / MB:.2f} MB"
            
        print(f"[ Cluster {cluster_id} ]")
        print(f"  -> Size:        {len(cluster_data)} users ({(len(cluster_data)/len(orig_df))*100:.1f}%)")
        print(f"  -> Peak Hour:   {peak_hour}")
        print(f"  -> Peak Volume: {vol_str} (Average per user in cluster)\n")
        
    print("="*50)

if __name__ == "__main__":
    main()
