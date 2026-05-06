import pandas as pd

def main():
    print("Loading clustered data...")
    try:
        df = pd.read_csv("clustered_telecom_dataset.csv")
    except FileNotFoundError:
        print("Error: 'clustered_telecom_dataset.csv' not found.")
        return
        
    # Define Time Windows
    night_cols = ['23:00', '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00']
    day_cols = [f"{str(i).zfill(2)}:00" for i in range(7, 23)]
    # Office hours: 09:00 to 18:00 inclusive (10 hours)
    office_cols = [f"{str(i).zfill(2)}:00" for i in range(9, 19)]
    all_cols = [f"{str(i).zfill(2)}:00" for i in range(24)]
    
    # Assumptions for Jio Add-on Pricing (in INR ₹)
    NIGHT_PACK_PRICE = 149
    WFH_PACK_PRICE = 251
    CONVERSION_RATE = 0.05
    
    summary = []
    
    for cluster_id in sorted(df['Cluster'].unique()):
        c_df = df[df['Cluster'] == cluster_id]
        size = len(c_df)
        
        # Calculate cluster averages for the specific time windows
        avg_night = c_df[night_cols].sum(axis=1).mean()
        avg_day = c_df[day_cols].sum(axis=1).mean()
        avg_office = c_df[office_cols].sum(axis=1).mean()
        avg_total = c_df[all_cols].sum(axis=1).mean()
        
        # Metrics
        night_to_day = avg_night / (avg_day + 1)
        office_pct = (avg_office / (avg_total + 1)) * 100
        
        # Logic Gate
        tag = "Standard User"
        recommendation = "Base Plan (No Add-on)"
        pack_price = 0
        
        if night_to_day > 2.0:
            tag = "Midnight Binger"
            recommendation = "Unlimited Night Data"
            pack_price = NIGHT_PACK_PRICE
        elif office_pct > 60.0:
            tag = "Remote Pro"
            recommendation = "50GB Work-from-Home"
            pack_price = WFH_PACK_PRICE
            
        # If cluster averages don't naturally trigger the rigid thresholds, 
        # let's do a smart fallback: check if a significant portion of *users* inside the cluster meet the criteria
        # to still assign a dominant tier.
        if tag == "Standard User":
            # Let's see if we should still classify it based on dominant user behavior in the cluster
            user_night_to_day = c_df[night_cols].sum(axis=1) / (c_df[day_cols].sum(axis=1) + 1)
            user_office_pct = (c_df[office_cols].sum(axis=1) / (c_df[all_cols].sum(axis=1) + 1)) * 100
            
            if (user_night_to_day > 2.0).mean() > 0.5:
                tag = "Midnight Binger"
                recommendation = "Unlimited Night Data"
                pack_price = NIGHT_PACK_PRICE
            elif (user_office_pct > 60.0).mean() > 0.3: # If >30% are heavy office users, we target the cluster
                tag = "Remote Pro"
                recommendation = "50GB Work-from-Home"
                pack_price = WFH_PACK_PRICE
                
        # Revenue Lift Math
        est_conversions = int(size * CONVERSION_RATE)
        revenue_lift = est_conversions * pack_price
        
        summary.append({
            'Cluster': f"Cluster {cluster_id}",
            'Population': size,
            'Avg Night/Day Ratio': f"{night_to_day:.2f}",
            'Avg Office %': f"{office_pct:.1f}%",
            'Jio Product Tier': tag,
            'Recommendation': recommendation,
            'Est. Conversions (5%)': est_conversions,
            'Revenue Lift (Rs.)': f"Rs.{revenue_lift:,}"
        })
        
    sum_df = pd.DataFrame(summary)
    
    print("\n" + "="*85)
    print("JIO PRODUCT TIER ASSIGNMENT & REVENUE LIFT PROJECTION")
    print("="*85)
    print(sum_df.to_string(index=False))
    print("="*85)
    
    total_lift = sum(int(row['Revenue Lift (Rs.)'].replace('Rs.','').replace(',','')) for row in summary)
    print(f"\n>> TOTAL PROJECTED REVENUE LIFT (MONTHLY): Rs.{total_lift:,}")

if __name__ == "__main__":
    main()
