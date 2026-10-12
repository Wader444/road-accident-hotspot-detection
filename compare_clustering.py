"""Compare the existing DBSCAN assignments with a reproducible K-Means model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


INPUT_FILE = Path("outputs/dbscan_results.csv")
OUTPUT_DIR = Path("outputs")
RESULTS_FILE = OUTPUT_DIR / "kmeans_results.csv"
FIGURE_FILE = OUTPUT_DIR / "clustering_comparison.png"
DEFAULT_K = 5
RANDOM_STATE = 42


def load_results(path: Path = INPUT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run clustering.py first to create DBSCAN results."
        )
    df = pd.read_csv(path)
    required = {"Start_Lat", "Start_Lng", "DBSCAN_Cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=list(required)).copy()
    if df.empty:
        raise ValueError("No valid records are available for clustering comparison.")
    return df


def project_to_kilometres(df: pd.DataFrame) -> np.ndarray:
    """Use a local equirectangular approximation around the study-area mean."""
    mean_lat = np.radians(df["Start_Lat"].mean())
    kilometres_per_degree = 111.32
    x = df["Start_Lng"].to_numpy() * np.cos(mean_lat) * kilometres_per_degree
    y = df["Start_Lat"].to_numpy() * kilometres_per_degree
    return np.column_stack((x, y))


def create_comparison(df: pd.DataFrame, k: int = DEFAULT_K) -> pd.DataFrame:
    if k < 2 or k > len(df):
        raise ValueError(f"K must be between 2 and the number of records ({len(df)}).")
    coordinates_km = project_to_kilometres(df)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["KMeans_Cluster"] = model.fit_predict(coordinates_km)
    return df


def save_comparison_chart(df: pd.DataFrame, path: Path = FIGURE_FILE) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    dbscan = df["DBSCAN_Cluster"]
    dbscan_colors = dbscan.map(lambda label: "#bdbdbd" if label == -1 else plt.cm.tab20(label % 20))
    axes[0].scatter(df["Start_Lng"], df["Start_Lat"], c=dbscan_colors, s=8, alpha=0.65)
    axes[0].set_title("DBSCAN (noise shown in grey)")
    axes[1].scatter(
        df["Start_Lng"], df["Start_Lat"], c=df["KMeans_Cluster"], cmap="tab10", s=8, alpha=0.65
    )
    axes[1].set_title("K-Means (K=5)")
    for axis in axes:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = create_comparison(load_results())
    df.to_csv(RESULTS_FILE, index=False)
    save_comparison_chart(df)

    dbscan_clusters = df.loc[df["DBSCAN_Cluster"] != -1, "DBSCAN_Cluster"].nunique()
    noise = int((df["DBSCAN_Cluster"] == -1).sum())
    print(f"Records compared: {len(df):,}")
    print(f"DBSCAN clusters: {dbscan_clusters:,}; noise records: {noise:,}")
    print(f"K-Means clusters: {df['KMeans_Cluster'].nunique():,}; K={DEFAULT_K}")
    print(f"Saved {RESULTS_FILE} and {FIGURE_FILE}")


if __name__ == "__main__":
    main()
