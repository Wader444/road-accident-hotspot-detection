
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "outputs/city_accidents.csv"
OUTPUT_DIR = "outputs"
CITY_MODE = os.environ.get("CITY_MODE", "0") == "1"

EARTH_RADIUS_M = 6_371_000

# Candidate parameters chosen after inspecting the sweep.
EPS_METERS = 400
MIN_SAMPLES = 5

# Parameters to compare during the sweep.
EPS_VALUES = [150, 250, 400, 600, 1000, 1500, 2500]
MIN_SAMPLES_VALUES = [5, 10, 15]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

print("\nLoading accident dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. "
        "Check that outputs/city_accidents.csv exists."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = ["City", "Start_Lat", "Start_Lng"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Start_Lat"] = pd.to_numeric(
    df["Start_Lat"], errors="coerce"
)
df["Start_Lng"] = pd.to_numeric(
    df["Start_Lng"], errors="coerce"
)

df = df.dropna(
    subset=["Start_Lat", "Start_Lng"]
).copy()

# Remove invalid latitude/longitude values.
df = df[
    df["Start_Lat"].between(-90, 90)
    & df["Start_Lng"].between(-180, 180)
].copy()

# The original Miami run uses an approximate rectangle. For a city selected
# from the raw dataset, the city filter in prepare_city.py is the study area.
if not CITY_MODE:
    df = df[
        df["Start_Lat"].between(25.0, 26.5)
        & df["Start_Lng"].between(-81.0, -79.5)
    ].copy()

df.reset_index(drop=True, inplace=True)

if len(df) < MIN_SAMPLES:
    raise ValueError(
        "Not enough valid records to perform DBSCAN."
    )

print(f"Records retained: {len(df)}")
print(
    "Study area: city-filtered records"
    if CITY_MODE
    else "Study area: approximate Miami-area bounds"
)

cleaned_name = "city_cleaned.csv" if CITY_MODE else "miami_cleaned.csv"
df.to_csv(os.path.join(OUTPUT_DIR, cleaned_name), index=False)


# ============================================================
# 3. PREPARE COORDINATES FOR HAVERSINE DISTANCE
# ============================================================

# Haversine expects [latitude, longitude] in radians.
coords_degrees = df[
    ["Start_Lat", "Start_Lng"]
].to_numpy()

coords_rad = np.radians(coords_degrees)


# ============================================================
# 4. GENERATE K-DISTANCE GRAPH
# ============================================================

print("\nGenerating k-distance graphs...")

K = 10

neighbors_model = NearestNeighbors(
    n_neighbors=K,
    metric="haversine",
    algorithm="ball_tree"
)

neighbors_model.fit(coords_rad)

distances_rad, _ = neighbors_model.kneighbors(coords_rad)

# DBSCAN counts the point itself as one of min_samples.
# Therefore the last returned neighbor here is the 10th
# neighbor when K = 10, including the point itself.
k_distances_m = np.sort(
    distances_rad[:, -1] * EARTH_RADIUS_M
)

pd.DataFrame({
    "k_distance_m": k_distances_m
}).to_csv(
    os.path.join(OUTPUT_DIR, "k_distances.csv"),
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(k_distances_m)
plt.xlabel("Points sorted by k-distance")
plt.ylabel("Distance to 10th nearest neighbor (metres)")
plt.title("DBSCAN k-distance graph")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "k_distance.png"),
    dpi=150
)
plt.close()

# Zoom the graph to make the main elbow easier to inspect.
zoom_limit = np.percentile(k_distances_m, 99)

plt.figure(figsize=(10, 6))
plt.plot(k_distances_m)
plt.ylim(0, max(zoom_limit, 1))
plt.xlabel("Points sorted by k-distance")
plt.ylabel("Distance to 10th nearest neighbor (metres)")
plt.title("Zoomed DBSCAN k-distance graph (99th percentile)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "k_distance_zoom.png"),
    dpi=150
)
plt.close()

print("Saved k-distance graphs.")


# ============================================================
# 5. DBSCAN PARAMETER SWEEP
# ============================================================

print("\nRunning DBSCAN parameter sweep...")

sweep_results = []

for min_samples in MIN_SAMPLES_VALUES:
    for eps_m in EPS_VALUES:

        eps_radians = eps_m / EARTH_RADIUS_M

        model = DBSCAN(
            eps=eps_radians,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree"
        )

        labels = model.fit_predict(coords_rad)

        cluster_ids = set(labels) - {-1}
        cluster_sizes = [
            int(np.sum(labels == cluster_id))
            for cluster_id in cluster_ids
        ]

        noise_records = int(np.sum(labels == -1))
        noise_percent = (
            noise_records / len(df) * 100
        )

        sweep_results.append({
            "eps_m": eps_m,
            "min_samples": min_samples,
            "clusters": len(cluster_ids),
            "noise_records": noise_records,
            "noise_percent": round(noise_percent, 2),
            "smallest_cluster": (
                min(cluster_sizes) if cluster_sizes else 0
            ),
            "largest_cluster": (
                max(cluster_sizes) if cluster_sizes else 0
            )
        })

sweep_df = pd.DataFrame(sweep_results)

sweep_path = os.path.join(
    OUTPUT_DIR, "dbscan_parameter_sweep.csv"
)

sweep_df.to_csv(sweep_path, index=False)

print("\n--- COMPLETE PARAMETER TABLE ---")
print(sweep_df.to_string(index=False))
print(f"\nSaved parameter sweep to: {sweep_path}")


# ============================================================
# 6. FIT THE SELECTED DBSCAN MODEL
# ============================================================

print("\nFitting selected DBSCAN model...")

eps_radians = EPS_METERS / EARTH_RADIUS_M

dbscan = DBSCAN(
    eps=eps_radians,
    min_samples=MIN_SAMPLES,
    metric="haversine",
    algorithm="ball_tree"
)

labels = dbscan.fit_predict(coords_rad)

df["DBSCAN_Cluster"] = labels

cluster_ids = sorted(
    set(labels) - {-1}
)

noise_count = int(np.sum(labels == -1))
cluster_count = len(cluster_ids)

print("\n========== SELECTED DBSCAN MODEL ==========")
print(f"Epsilon: {EPS_METERS} metres")
print(f"Minimum samples: {MIN_SAMPLES}")
print(f"Total records: {len(df)}")
print(f"Number of clusters: {cluster_count}")
print(f"Noise records: {noise_count}")
print(
    f"Noise percentage: "
    f"{noise_count / len(df) * 100:.2f}%"
)


# ============================================================
# 7. ANALYZE CLUSTER SIZES
# ============================================================

clustered_df = df[
    df["DBSCAN_Cluster"] != -1
].copy()

cluster_sizes = (
    clustered_df
    .groupby("DBSCAN_Cluster")
    .size()
    .sort_values(ascending=False)
)

cluster_sizes_df = cluster_sizes.rename(
    "accident_records"
).reset_index()

cluster_sizes_df.to_csv(
    os.path.join(OUTPUT_DIR, "dbscan_cluster_sizes.csv"),
    index=False
)

print("\n--- TOP 10 LARGEST CLUSTERS ---")

if not cluster_sizes_df.empty:
    print(
        cluster_sizes_df.head(10).to_string(index=False)
    )
else:
    print("No clusters were detected.")


# ============================================================
# 8. CALCULATE HOTSPOT CENTERS
# ============================================================

if not clustered_df.empty:

    hotspot_centers = (
        clustered_df
        .groupby("DBSCAN_Cluster")
        .agg(
            accident_records=("Start_Lat", "size"),
            center_lat=("Start_Lat", "mean"),
            center_lng=("Start_Lng", "mean")
        )
        .reset_index()
        .sort_values(
            "accident_records",
            ascending=False
        )
    )

    hotspot_centers.to_csv(
        os.path.join(OUTPUT_DIR, "dbscan_hotspot_centers.csv"),
        index=False
    )

    print("\n--- TOP 10 HOTSPOT CENTERS ---")
    print(
        hotspot_centers.head(10).to_string(index=False)
    )

else:
    hotspot_centers = pd.DataFrame(
        columns=[
            "DBSCAN_Cluster",
            "accident_records",
            "center_lat",
            "center_lng"
        ]
    )


# ============================================================
# 9. SAVE FINAL RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR, "dbscan_results.csv"
)

df.to_csv(results_path, index=False)

print("\n========== OUTPUT FILES ==========")
print("outputs/miami_cleaned.csv")
print("outputs/k_distance.png")
print("outputs/k_distance_zoom.png")
print("outputs/k_distances.csv")
print("outputs/dbscan_parameter_sweep.csv")
print("outputs/dbscan_cluster_sizes.csv")
print("outputs/dbscan_hotspot_centers.csv")
print("outputs/dbscan_results.csv")

print("\nDBSCAN execution completed successfully!")
