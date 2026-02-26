import pandas as pd
from core.data_loader import load_full_data


def calculate_rfm():
    df = load_full_data()

    # Drop missing customers
    df = df.dropna(subset = ["CustomerID"])

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Snapshot date = day after last transaction
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days = 1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"  
        }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    return rfm

def add_rfm_scores(rfm):
    rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])

    rfm["R_score"] = rfm["R_score"].astype(int)
    rfm["F_score"] = rfm["F_score"].astype(int)
    rfm["M_score"] = rfm["M_score"].astype(int)

    rfm["RFM_score"] = (
        rfm["R_score"].astype(str) +
        rfm["F_score"].astype(str) +
        rfm["M_score"].astype(str)
        )

    return rfm

def assign_segments(rfm):

    def segment(row):
        if row["R_score"] == 4 and row["F_score"] >= 3:
            return "VIP"
        elif row["R_score"] >= 3 and row["F_score"] >= 2:
            return "Loyal"
        elif row["R_score"] == 1:
            return "At risk"
        elif row["F_score"] == 1:
            return "Occasional"
        else:
            return "Regular"

    rfm["Segment"] = rfm.apply(segment, axis=1)

    return rfm

if __name__ == "__main__":
    rfm = calculate_rfm()
    rfm = add_rfm_scores(rfm)
    rfm = assign_segments(rfm)

    print("\nRFM")
    print(rfm.head())

    print("\nSEGMENT DISTRIBUTION")
    print(rfm["Segment"].value_counts())