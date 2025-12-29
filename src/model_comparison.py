import pandas as pd
import numpy as np
from clustering_library import DataCleaner, FeatureEngineer, ClusterAnalyzer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def run_comparison():
    print("="*80)
    print("CUSTOMER SEGMENTATION MODEL COMPARISON REPORT")
    print("="*80)

    # 1. Load Data
    print("\n[1] Loading Data...")
    cleaner = DataCleaner("../data/raw/data.csv") # Assuming raw exists, or we use processed
    # Safe load
    try:
        analyzer = ClusterAnalyzer()
        df_scaled, df_original = analyzer.load_data()
        print("    Data loaded successfully.")
    except Exception as e:
        print(f"    Error loading data: {e}")
        return

    results = []

    # 2. K-Means (Assuming k=4 as an optimal choice from previous context)
    print("\n[2] Running K-Means (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(df_scaled)
    metrics_km = analyzer.evaluate_clustering(kmeans_labels)
    results.append({
        "Model": "K-Means (k=4)",
        **metrics_km
    })

    # 3. DBSCAN (Assuming eps=0.5, min_samples=5 as starting point or optimized)
    # Note: DBSCAN often needs tuning. I'll use parameters likely to give results.
    print("\n[3] Running DBSCAN...")
    dbscan = DBSCAN(eps=2.0, min_samples=5) # Adjusted eps for scaled data
    dbscan_labels = dbscan.fit_predict(df_scaled)
    # DBSCAN might produce only -1 if not tuned well.
    if len(set(dbscan_labels)) > 1:
        metrics_db = analyzer.evaluate_clustering(dbscan_labels)
    else:
        metrics_db = {"Silhouette Score": -1, "Davies-Bouldin Index": -1, "Calinski-Harabasz Score": -1}
    
    results.append({
        "Model": "DBSCAN",
        **metrics_db
    })

    # 4. Hierarchical (k=4)
    print("\n[4] Running Hierarchical Clustering (k=4)...")
    agg = AgglomerativeClustering(n_clusters=4)
    hier_labels = agg.fit_predict(df_scaled)
    metrics_hc = analyzer.evaluate_clustering(hier_labels)
    results.append({
        "Model": "Hierarchical (k=4)",
        **metrics_hc
    })

    # 5. Comparison Table
    print("\n" + "="*80)
    print("FINAL COMPARATIVE ANALYSIS")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # 6. Recommendation
    best_model = results_df.loc[results_df['Silhouette Score'].idxmax()]
    print("\n" + "-"*80)
    print(f"RECOMMENDATION: Based on Silhouette Score, the best model is {best_model['Model']}.")
    print("-"*80)

if __name__ == "__main__":
    run_comparison()
