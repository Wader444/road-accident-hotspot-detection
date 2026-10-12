# Road Accident Hotspot Detection Using Machine Learning

## Objective

This project uses geographic accident coordinates to identify dense accident
hotspots in an approximate Miami-area study region. DBSCAN is the primary
algorithm because it can discover clusters without requiring a fixed number of
clusters and can label isolated records as noise. K-Means is included as a
comparison model.

## Problem statement and motivation

Accident records contain locations, but a long list of coordinates is difficult
to interpret. Clustering converts those records into groups that can be mapped
and summarized. The result describes concentrations of recorded accidents; it
does **not** estimate per-vehicle risk because this project has no traffic
volume, road-usage, or exposure denominator.

## Dataset and current columns

The current project input is `outputs/city_accidents.csv`, a prepared subset
with the columns `City`, `Start_Lat`, and `Start_Lng`. The raw dataset and its
provenance are not present in this workspace, so no additional source details
are claimed here. The DBSCAN result file adds `DBSCAN_Cluster`. Optional fields
such as severity, date, weather, and road conditions were checked and are not
available in the current subset.

The existing pipeline filters valid coordinates to approximate bounds
25.0--26.5 latitude and -81.0--79.5 longitude, then uses DBSCAN with a
400-metre epsilon and `min_samples=5`.

## Folder structure

```text
ML-CEP/
├── clustering.py
├── analyze_hotspots.py
├── visualize_hotspots.py
├── compare_clustering.py
├── main.py
├── diagnose_data.py
├── requirements.txt
├── data/                         # optional raw input location
└── outputs/                      # input and generated results
```

## Setup on Windows

Python 3.13 is the tested environment for this workspace.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the scripts directly with
`.venv\Scripts\python.exe`.

## Running the project

Run the existing DBSCAN pipeline only when the input data or DBSCAN settings
need to be regenerated:

```powershell
python clustering.py
```

To analyze another city from the raw US Accidents file, use the city runner.
The raw file is read in chunks, so the complete 3 GB file is not loaded into
memory:

```powershell
python run_city.py "Orlando" --max-rows 10000
```

Replace `Orlando` with another city name present in the raw dataset. This
command prepares the city, runs DBSCAN, regenerates the summaries, and creates
the comparison outputs. It replaces the current files in `outputs/`, so copy
any Miami outputs you want to keep before running another city.

To only prepare a city without running clustering:

```powershell
python prepare_city.py "Orlando" --max-rows 10000
```

Run each new stage independently:

```powershell
python analyze_hotspots.py
python visualize_hotspots.py
python compare_clustering.py
```

The easiest way to generate and open the map is:

```powershell
python run_map.py
```

This starts a local server at `http://127.0.0.1:8000`, opens the map in your
browser, and keeps the server running until you press `Ctrl+C`. Serving the
HTML file locally is more reliable than opening a dense map directly with a
`file://` URL.

Run all downstream stages without repeating the expensive DBSCAN parameter
sweep:

```powershell
python main.py
```

Open `outputs\accident_hotspots_map.html` in a web browser to inspect the
interactive map.

## Concepts

### DBSCAN and Haversine distance

DBSCAN groups points that have enough neighbours within a distance threshold.
Here coordinates are converted to radians and DBSCAN uses the Haversine metric,
which measures distance along the Earth's surface. The selected `eps` is 400
metres. `min_samples=5` means a dense core point needs at least five records
within that neighbourhood, including itself. Records that do not belong to a
dense region receive label `-1` and are called noise.

The selected values are a candidate configuration, not proof of an optimal
model. The existing parameter sweep can be inspected in
`outputs/dbscan_parameter_sweep.csv`.

### K-Means comparison

K-Means assigns every record to one of a fixed number of centroids. This
project uses `K=5` and `random_state=42`. Since K-Means uses Euclidean
distance, latitude and longitude are converted to a local kilometre
approximation before fitting. Cluster numbers are labels only; their numeric
values have no geographic meaning. K-Means does not provide a noise label,
while DBSCAN can identify isolated records.

## Generated outputs

| File | Description |
|---|---|
| `miami_cleaned.csv` | Cleaned study-area records |
| `dbscan_results.csv` | Accident records with DBSCAN labels |
| `dbscan_parameter_sweep.csv` | Candidate DBSCAN settings and counts |
| `dbscan_cluster_sizes.csv` | Existing DBSCAN cluster-size export |
| `dbscan_hotspot_centers.csv` | Existing mean center for each DBSCAN cluster |
| `hotspot_summary.csv` | Ranked cluster sizes and center coordinates |
| `top_10_hotspots.png` | Bar chart of the ten largest hotspots |
| `accident_hotspots_map.html` | Interactive Folium map |
| `kmeans_results.csv` | Same records with K-Means labels |
| `clustering_comparison.png` | DBSCAN/K-Means side-by-side plot |

## Observed results in the current outputs

The existing `dbscan_results.csv` contains 3,625 valid records. With the
candidate 400-metre and five-record settings, the saved parameter sweep shows
73 clusters and 668 noise records (18.43%). The largest saved cluster contains
916 records. These are observations from the current files, not guaranteed
results for a different dataset or rerun.

## Limitations and future improvements

* Accident counts are not exposure-adjusted risk.
* The study area is an approximate geographic rectangle, not an official
  municipal boundary.
* The current subset has no severity or temporal fields for additional
  analysis.
* DBSCAN results depend on `eps` and `min_samples`; sensitivity analysis should
  be considered before operational use.
* Future work could add road-network data, traffic exposure, severity, time,
  weather, and statistically validated spatial methods.

## Demonstration checklist

1. Activate the virtual environment.
2. Run `python main.py`.
3. Show the printed record, cluster, and noise summary from
   `python analyze_hotspots.py`.
4. Open `outputs\accident_hotspots_map.html`; toggle accident and hotspot-center
   layers and click markers.
5. Show `outputs\top_10_hotspots.png` and
   `outputs\clustering_comparison.png`.
6. Explain that DBSCAN finds density and noise, while K-Means assigns all
   records to five predefined groups.

## Likely viva questions

**Why use DBSCAN?** It discovers dense regions without choosing the number of
clusters in advance and can mark isolated observations as noise.

**Why convert coordinates to radians?** Scikit-learn's Haversine metric expects
latitude and longitude in radians.

**What does epsilon mean?** It is the neighbourhood radius; here it represents
400 metres after conversion using the approximate Earth radius.

**What does noise mean?** A record labelled `-1` did not satisfy the selected
density requirements. It is not necessarily an invalid accident.

**Why compare K-Means?** It provides a contrasting centroid-based method with a
fixed K, illustrating how algorithm assumptions change the grouping.

**Does the largest cluster prove the highest accident risk?** No. It shows the
largest concentration of recorded accidents in this dataset. Risk requires
exposure information such as traffic volume or vehicle-kilometres.

**Can cluster IDs be compared between algorithms?** No. Numeric cluster IDs are
arbitrary labels; compare spatial patterns and size distributions instead.