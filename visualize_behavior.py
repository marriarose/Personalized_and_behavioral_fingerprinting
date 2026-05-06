import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def main():
    # 1. Minimal Dark-Mode Aesthetics
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    # Background colors
    bg_color = '#1e1e2e'  # Soft dark background
    
    # 2. Load Data
    print("Loading data...")
    raw_df = pd.read_csv("clustered_telecom_dataset.csv")
    scaled_features = pd.read_csv("scaled_engineered_features.csv")
    
    # Map clusters to scaled features
    scaled_features['Cluster'] = raw_df['Cluster']
    
    # 3. Prepare Heatmap Data (X=Hours, Y=Clusters)
    hour_cols = [f"{str(i).zfill(2)}:00" for i in range(24)]
    heatmap_data = raw_df.groupby('Cluster')[hour_cols].mean()
    
    # Normalize per cluster (row-wise) so intensity is relative to its own peak
    # This prevents the Heavy Streamers from completely washing out the Commuters' colors
    heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    
    # 4. Prepare Radar Chart Data (Centroids of the 4 features)
    feature_cols = [
        'Night-to-Day Ratio', 
        'Weekend Peak Intensity', 
        'Consistency Score', 
        'Prime-Time Concentration'
    ]
    radar_data = scaled_features.groupby('Cluster')[feature_cols].mean().reset_index()
    
    # Min-Max scaling for the radar chart specifically (since robust scaler gave negative values)
    for col in feature_cols:
        c_min = radar_data[col].min()
        c_max = radar_data[col].max()
        if c_max - c_min != 0:
            radar_data[col] = (radar_data[col] - c_min) / (c_max - c_min)
        else:
            radar_data[col] = 0.5
            
    # Radar Math setup
    categories = feature_cols
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # 5. Create Figure
    fig = plt.figure(figsize=(18, 7), facecolor=bg_color)
    
    # --- SUBPLOT 1: Behavioral Heatmap ---
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor(bg_color)
    sns.heatmap(
        heatmap_data_norm, 
        cmap='YlGnBu', 
        ax=ax1, 
        cbar_kws={'label': 'Normalized Peak Intensity'},
        linewidths=0.5,
        linecolor=bg_color
    )
    ax1.set_title('Hourly Behavioral Heatmap', color='white', pad=20, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cluster Labels', color='white', fontsize=12)
    ax1.set_xlabel('Hour of Day', color='white', fontsize=12)
    ax1.tick_params(colors='white')
    
    # Format x-ticks to just show the hour number for cleaner look
    x_labels = [str(i).zfill(2) for i in range(24)]
    ax1.set_xticklabels(x_labels, rotation=0)
    
    # --- SUBPLOT 2: Radar Chart ---
    ax2 = plt.subplot(1, 2, 2, polar=True)
    ax2.set_facecolor(bg_color)
    
    cluster_colors = ['#00ffcc', '#ff3366'] # Neon Cyan and Neon Pink
    
    for i, row in radar_data.iterrows():
        values = row[feature_cols].values.flatten().tolist()
        values += values[:1] # Close the loop
        
        cluster_name = "Cluster 0 (Commuters)" if row["Cluster"] == 0 else "Cluster 1 (Night-Owls)"
        
        ax2.plot(angles, values, linewidth=2.5, linestyle='solid', label=cluster_name, color=cluster_colors[i % len(cluster_colors)])
        ax2.fill(angles, values, color=cluster_colors[i % len(cluster_colors)], alpha=0.15)
        
    # Aesthetics: No gridlines
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, color='white', size=11, fontweight='bold')
    
    ax2.set_yticklabels([]) # Hide radial labels
    ax2.grid(False) # NO GRIDLINES
    ax2.spines['polar'].set_visible(False) # Remove outer ring
    
    ax2.set_title("Average User Signature (Engineered Features)", color='white', pad=30, fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=bg_color, edgecolor='none', labelcolor='white', fontsize=11)
    
    # 6. Save Plot
    plt.tight_layout()
    output_image = 'behavioral_visualization.png'
    plt.savefig(output_image, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    print(f"Visualization successfully generated and saved to '{output_image}'.")

if __name__ == "__main__":
    main()
