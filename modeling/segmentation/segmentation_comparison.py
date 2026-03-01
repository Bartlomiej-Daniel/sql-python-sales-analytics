import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.sql.operators import comparison_op
from modeling.segmentation.kmeans_segmentation import (
    prepare_data,
    apply_kmeans
)

def compare_segments():
    # RFM
    rfm, scaled_data = prepare_data()

    # K-Means 
    rfm, model = apply_kmeans(rfm, scaled_data, n_cluster=3)
    comparison = pd.crosstab(rfm["Cluster"], rfm["Segment"])

    # Crosstab
    comparison = pd.crosstab(rfm["Cluster"], rfm["Segment"])

    print("\nRFM vs KMeans Comparison:")
    print(comparison)
    
    return comparison

def plot_heatmap(comparison):
    plt.figure(figsize=(8,6))
    sns.heatmap(comparison, annot = True, fmt = "d", cmap = "Blues")
    plt.title("RFM vs KMeans Segment Comparison")
    plt.ylabel("KMeans Cluster")
    plt.xlabel("RFM Segment")
    plt.tight_layout()

    plt.savefig("reports/RFM_vs_KMeans_Segment_Comparison.png")
    plt.close()

if __name__ == "__main__":
    comparison = compare_segments()
    plot_heatmap(comparison)
