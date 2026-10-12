"""Summarize DBSCAN hotspot sizes and create an analytical chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("outputs/dbscan_results.csv")
OUTPUT_DIR = Path("outputs")
SUMMARY_FILE = OUTPUT_DIR / "hotspot_summary.csv"
CHART_FILE = OUTPUT_DIR / "top_10_hotspots.png"
REQUIRED_COLUMNS = {"Start_Lat", "Start_Lng", "DBSCAN_Cluster"}


def load_results(path: Path = INPUT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run clustering.py first to create DBSCAN results."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for column in ("Start_Lat", "Start_Lng", "DBSCAN_Cluster"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=list(REQUIRED_COLUMNS)).copy()


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["DBSCAN_Cluster"] != -1].copy()
    summary = (
        clustered.groupby("DBSCAN_Cluster")
        .agg(
            accident_records=("DBSCAN_Cluster", "size"),
            center_lat=("Start_Lat", "mean"),
            center_lng=("Start_Lng", "mean"),
        )
        .reset_index()
        .sort_values(["accident_records", "DBSCAN_Cluster"], ascending=[False, True])
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def save_chart(summary: pd.DataFrame, path: Path = CHART_FILE) -> None:
    top = summary.head(10).sort_values("accident_records")
    figure, axis = plt.subplots(figsize=(9, 6))
    if top.empty:
        axis.text(0.5, 0.5, "No DBSCAN clusters detected", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.barh(top["DBSCAN_Cluster"].astype(str), top["accident_records"], color="#2c7fb8")
        axis.set_xlabel("Number of accident records")
        axis.set_ylabel("DBSCAN cluster label")
        axis.set_title("Ten largest DBSCAN accident hotspots")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    summary = analyze(df)
    summary.to_csv(SUMMARY_FILE, index=False)
    save_chart(summary)

    noise = int((df["DBSCAN_Cluster"] == -1).sum())
    print(f"Valid accident records: {len(df):,}")
    print(f"Detected clusters: {len(summary):,}")
    print(f"Noise records: {noise:,} ({noise / len(df) * 100:.2f}%)")
    print("\nLargest clusters:")
    print(summary.head(10).to_string(index=False))
    print(f"\nSaved {SUMMARY_FILE} and {CHART_FILE}")


if __name__ == "__main__":
    main()
