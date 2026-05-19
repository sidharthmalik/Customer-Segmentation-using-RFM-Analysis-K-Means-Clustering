import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "online_retail_ii.xlsx")
OUTPUT_DIR = "output"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Invoice", "StockCode", "Customer ID", "Description"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum",
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm = rfm[rfm["Monetary"] > 0]
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["RScore"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["FScore"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["MScore"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFMScore"] = rfm["RScore"].astype(str) + rfm["FScore"].astype(str) + rfm["MScore"].astype(str)
    rfm["RFMTotal"] = rfm[["RScore", "FScore", "MScore"]].sum(axis=1)
    return rfm


def build_cluster_features(rfm: pd.DataFrame) -> pd.DataFrame:
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled, index=features.index, columns=features.columns)
    return scaled_features, scaler


def choose_k(features: pd.DataFrame, max_k: int = 8) -> int:
    inertias = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(features)
        inertias.append(model.inertia_)
    diffs = np.diff(inertias)
    elbow_k = np.argmin(diffs) + 2
    return int(elbow_k)


def apply_kmeans(features: pd.DataFrame, n_clusters: int) -> pd.Series:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(features)
    return pd.Series(labels, index=features.index, name="Cluster")


def profile_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    return rfm.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": ["mean", "sum"],
        "RFMTotal": "mean",
    })


def save_outputs(rfm: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_segments.csv"), index=True)
    cluster_summary = profile_clusters(rfm)
    cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_profile.csv"))
    print(f"Saved analysis outputs to {OUTPUT_DIR}/")


def main() -> None:
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    df = clean_data(df)
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    features, _scaler = build_cluster_features(rfm)
    k = choose_k(features)
    rfm["Cluster"] = apply_kmeans(features, k)
    save_outputs(rfm)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
