import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from modeling.segmentation.rfm import calculate_rfm, add_rfm_scores, assign_segments


def prepare_data():
    rfm = calculate_rfm()
    rfm = add_rfm_scores(rfm)
    rfm = assign_segments(rfm)

    features = rfm[["Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return rfm, scaled_features

def find_optimal_clusters(data):
    inertias = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(6,4))
    plt.plot(range(1,11), inertias, marker="o")
    plt.title("Elbow method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()

def apply_kmeans(rfm, data, n_cluster=4):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)

    rfm["Cluster"] = clusters
    
    return rfm, kmeans

def plot_clusters(rfm, save_dir="reports", show=False):

    # Recency vs Monetary
    plt.figure(figsize=(8,6))
    plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"])
    plt.yscale("log")
    plt.xlabel("Recency (days)")
    plt.ylabel("Monetary (log scale)")
    plt.title("Clusters: Recency vs Monetary")
    plt.tight_layout()
    plt.savefig("reports/clusters_recency_monetary.png")
    
    if show:
        plt.show()
    plt.close()


    # Frequency vs Monetary
    plt.figure(figsize=(8,6))
    plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
    plt.yscale("log")
    plt.xlabel("Frequency")
    plt.ylabel("Monetary (log scale)")
    plt.title("Clusters: Frequency vs Monetary")
    plt.tight_layout()
    plt.savefig("reports/clusters_frequency_monetary.png")

    if show:
        plt.show()
    plt.close()

    print("Cluster plots saved in reports/")

def plot_revenue_share(rfm):
    revenue_cluster = (
        rfm.groupby("Cluster")["Monetary"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_share = revenue_cluster / revenue_cluster.sum() * 100

    plt.figure(figsize=(8,6))
    revenue_share.plot(kind="bar")
    plt.ylabel("Revenue Share (%)")
    plt.title("Revenue Share by Cluster")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/cluster_revenue_share.png")
    plt.close()

    print("\nRevenue share by cluster:")
    print(revenue_share.round(2))
    print("\nRevenue share plot saved in reports/")

def calculate_silhouette(scaled_data, labels):
    score = silhouette_score(scaled_data, labels)

    print("\nSilhouette Score:")
    print(round(score, 4))

    return score


if __name__ == "__main__":
    rfm, scaled_data = prepare_data()

    print("Finding optimal clusters")
    find_optimal_clusters(scaled_data)

    rfm, model = apply_kmeans(rfm, scaled_data, n_cluster=3)

    calculate_silhouette(scaled_data, rfm["Cluster"])

    print("\nCluster distribution")
    print(rfm["Cluster"].value_counts())

    print("\nCluster vs Business Segment:")
    print(pd.crosstab(rfm["Cluster"], rfm["Segment"]))

    print("\nCluster Profile (Mean RFM values):")
    cluster_profile = (
        rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
    )

    print(cluster_profile)
    
    plot_clusters(rfm)

    plot_revenue_share(rfm)

    


