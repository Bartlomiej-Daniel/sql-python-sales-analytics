from re import S
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # VIP analysis
    vip_total =len(rfm[rfm["Segment"] == "VIP"])
    vip_in_cluster2 = len(rfm[(rfm["Segment"] == "VIP") & (rfm["Cluster"] == 2)])

    vip_capture_rate = vip_in_cluster2 / vip_total * 100

    cluster2_total = len(rfm[rfm["Cluster"] == 2])
    cluster2_vip_share = vip_in_cluster2 / cluster2_total * 100

    print("\nVIP total:", vip_total)
    print("VIP in Cluster 2:", vip_in_cluster2)
    print("VIP capture rate:", round(vip_capture_rate, 2), "%")
    print("VIP share inside Cluster 2:", round(cluster2_vip_share, 2), "%")

    # Revenue analysis
    total_revenue = rfm["Monetary"].sum()

    cluster2_revenue = rfm[rfm["Cluster"] == 2]["Monetary"].sum()
    cluster2_revenue_share = cluster2_revenue / total_revenue * 100

    vip_revenue = rfm[rfm["Segment"] == "VIP"]["Monetary"].sum()

    # Average revenue
    cluster2_avg = rfm[rfm["Cluster"] == 2]["Monetary"].mean()
    vip_avg = rfm[rfm["Segment"] == "VIP"]["Monetary"].mean()

    print("\nRevenue Deep Dive")
    print("Total revenue: ", round(total_revenue, 2))
    print("Cluster 2 revenue: ", round(cluster2_revenue, 2))
    print("Cluster revenue share: ", round(cluster2_revenue_share, 2))

    print("\nVIP total revenue: ", round(vip_revenue,2))

    print("\nAverage revenue per customer: ")
    print("Cluster 2 avg: ", round(cluster2_avg,2))
    print("VIP avg: ", round(vip_avg,2))

    multiple = cluster2_avg / vip_avg
    print("\nUltra VIP vs Average VIP multiple:", round(multiple, 2), "x")

    
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
