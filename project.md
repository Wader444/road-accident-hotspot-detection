# Road Accident Hotspot Detection Using Machine Learning

## 1. Project at a glance

### Project title

**Road Accident Hotspot Detection Using Machine Learning**

### Main objective

This project analyzes accident coordinates from an approximate Miami-area
dataset. It uses geographic clustering to identify areas where many recorded
accidents are concentrated, displays those areas on an interactive map, and
compares DBSCAN with K-Means.

The project identifies **accident concentrations**, not guaranteed accident
risk. A true risk calculation would also require exposure information such as
traffic volume, road usage, or vehicle-kilometres travelled.

### Main technologies

- Python
- pandas: data loading, cleaning, grouping, and CSV export
- NumPy: numerical and coordinate calculations
- scikit-learn: DBSCAN, Nearest Neighbours, and K-Means
- Matplotlib: static charts
- Folium: interactive browser map

---

## 2. End-to-end workflow

The project follows this flow:

```text
Accident data
    |
    v
diagnose_data.py
    |
    v
clustering.py
    |
    v
dbscan_results.csv
    |
    +--> analyze_hotspots.py ------> hotspot_summary.csv
    |                                top_10_hotspots.png
    |
    +--> visualize_hotspots.py ----> accident_hotspots_map.html
    |
    +--> compare_clustering.py ----> kmeans_results.csv
                                     clustering_comparison.png
    |
    v
main.py runs the downstream stages together
```

The expensive DBSCAN parameter sweep is performed by `clustering.py`.
`main.py` does not repeat that sweep; it uses the existing DBSCAN output to
regenerate analysis, maps, and comparison charts.

---

## 3. Explanation of every source file

### `clustering.py`

**Purpose:** Main DBSCAN processing pipeline.

**What it does:**

1. Loads `outputs/city_accidents.csv`.
2. Checks that `City`, `Start_Lat`, and `Start_Lng` exist.
3. Converts latitude and longitude values to numeric values.
4. Removes missing or geographically invalid coordinates.
5. Applies the approximate Miami-area bounds:
   - Latitude: `25.0` to `26.5`
   - Longitude: `-81.0` to `-79.5`
6. Saves the cleaned data to `outputs/miami_cleaned.csv`.
7. Converts coordinates from degrees to radians.
8. Calculates nearest-neighbour distances for a k-distance graph.
9. Runs a DBSCAN parameter sweep.
10. Fits the selected DBSCAN model.
11. Calculates cluster sizes and average cluster coordinates.
12. Saves each accident with its DBSCAN label.

**Important settings:**

```python
EPS_METERS = 400
MIN_SAMPLES = 5
EARTH_RADIUS_M = 6_371_000
```

**Fundamental concept:** This is the unsupervised learning stage. There is no
target column such as `hotspot = yes/no`; the algorithm discovers spatial
patterns directly from the coordinates.


### `diagnose_data.py`

**Purpose:** Inspect the prepared input before clustering.

**What it does:**

- Prints the dataset shape.
- Lists available columns.
- Checks coordinate ranges.
- Reports summary statistics.
- Displays city counts.
- Reports records outside the approximate study bounds.

**Fundamental concept:** Data profiling should happen before model training.
It helps reveal missing fields, invalid coordinates, unexpected cities, and
outliers before they affect the analysis.

### `visualize_hotspots.py`

**Purpose:** Create the interactive accident hotspot map.

**What it does:**

- Loads `outputs/dbscan_results.csv`.
- Uses the existing `DBSCAN_Cluster` column.
- Shows clustered records with distinguishable colors.
- Shows noise records, label `-1`, in grey.
- Groups markers to keep the map usable.
- Adds popup information such as cluster ID, latitude, longitude, and any
  additional available fields.
- Adds hotspot-center markers when
  `outputs/dbscan_hotspot_centers.csv` is available.
- Adds a legend and layer control.

**Output:**

```text
outputs/accident_hotspots_map.html
```

**Fundamental concept:** Visualization converts model output into an
interpretable result. A cluster label is difficult to understand in a CSV but
becomes meaningful when displayed geographically.

### `analyze_hotspots.py`

**Purpose:** Calculate and export statistical summaries of the DBSCAN output.

**What it does:**

- Counts valid accident records.
- Counts detected clusters.
- Counts noise records.
- Calculates the noise percentage.
- Ranks clusters by accident-record count.
- Calculates the mean latitude and longitude for each cluster.
- Creates the top-ten hotspot chart.

**Outputs:**

```text
outputs/hotspot_summary.csv
outputs/top_10_hotspots.png
```

**Fundamental concept:** Exploratory data analysis summarizes the model
results. The cluster size is a count of recorded accidents, not a direct
measure of per-driver or per-vehicle accident probability.

### `compare_clustering.py`

**Purpose:** Compare DBSCAN with K-Means using the same records.

**What it does:**

- Loads the DBSCAN results.
- Projects latitude and longitude into a local kilometre approximation.
- Fits K-Means with `K=5`.
- Uses `random_state=42` for reproducibility.
- Saves K-Means labels.
- Creates a side-by-side DBSCAN and K-Means plot.

**Outputs:**

```text
outputs/kmeans_results.csv
outputs/clustering_comparison.png
```

**Fundamental concept:** Different clustering algorithms make different
assumptions:

- DBSCAN identifies dense areas and can label noise.
- K-Means creates a fixed number of centroid-based groups.
- K-Means assigns every record to a cluster.
- Cluster numbers are arbitrary labels and must not be compared by their
  numeric values.

### `main.py`

**Purpose:** Provide a single entry point for the completed downstream
workflow.

**What it does:**

1. Checks that `outputs/dbscan_results.csv` exists.
2. Runs hotspot analysis.
3. Generates the interactive map.
4. Runs the K-Means comparison.
5. Prints the generated files.

**Fundamental concept:** Modular pipeline orchestration. Each stage can be
debugged independently, while `main.py` provides a convenient reproducible
demonstration command.

### `run_map.py`

**Purpose:** Provide the easiest way to view the dense Folium map.

**What it does:**

1. Regenerates `outputs/accident_hotspots_map.html`.
2. Starts a local HTTP server on `127.0.0.1:8000`.
3. Opens the map in the default browser.
4. Keeps the server alive until `Ctrl+C` is pressed.

**Why it is useful:** A dense HTML map can be slow or unreliable when opened
directly using a `file://` URL. A local HTTP server serves the same file in a
normal browser context and is the recommended viewing method.

Run it from the project folder:

```powershell
python run_map.py
```

Then keep the terminal open while using the map. Press `Ctrl+C` when finished.

### `prepare_city.py`

**Purpose:** Extract records for a selected city from the raw
`data/US_Accidents_March23.csv` file.

**What it does:**

- Reads the large raw CSV in chunks.
- Matches the `City` column case-insensitively.
- Keeps valid `Start_Lat` and `Start_Lng` values.
- Saves the selected city to `outputs/city_accidents.csv`.

Run:

```powershell
python prepare_city.py "Orlando" --max-rows 10000
```

The `--max-rows` value is a reproducible processing limit. Use a smaller
number for a quick experiment or a larger number when enough records are
available.

### `run_city.py`

**Purpose:** Run the complete workflow for a city in one command.

Run:

```powershell
python run_city.py "Orlando" --max-rows 10000
```

It prepares the city, runs DBSCAN without the Miami-specific rectangle,
generates the statistical analysis, creates the K-Means comparison, and
refreshes the downstream outputs. The generated files in `outputs/` represent
the most recently selected city, so save or rename results before processing a
different city.

### `requirements.txt`

**Purpose:** Lists the Python packages required by the project.

Current dependencies:

```text
pandas
numpy
scikit-learn
matplotlib
folium
```

**Fundamental concept:** Dependency management makes the project easier to
install and reproduce on another computer.

### `README.md`

**Purpose:** General project documentation.

It contains setup instructions, commands, concepts, outputs, limitations,
observed results, and basic viva questions.

### `project.md`

**Purpose:** Detailed project explanation and evaluator presentation guide.

This file explains the role and concept of every project file and provides a
step-by-step way to demonstrate the project.

---

## 4. Explanation of the main output files

### Input and cleaning outputs

| File | Meaning |
|---|---|
| `city_accidents.csv` | Prepared accident records used as the starting input |
| `miami_cleaned.csv` | Records after numeric, geographic, and study-area filtering |

### DBSCAN outputs

| File | Meaning |
|---|---|
| `k_distances.csv` | Sorted nearest-neighbour distances used to inspect density |
| `k_distance.png` | Full k-distance graph |
| `k_distance_zoom.png` | Zoomed k-distance graph |
| `dbscan_parameter_sweep.csv` | Results for multiple `eps` and `min_samples` combinations |
| `dbscan_results.csv` | Every accident record with its DBSCAN label |
| `dbscan_cluster_sizes.csv` | Size of every non-noise cluster |
| `dbscan_hotspot_centers.csv` | Mean coordinate of every non-noise cluster |

### Analysis and presentation outputs

| File | Meaning |
|---|---|
| `hotspot_summary.csv` | Ranked hotspot summary with sizes and center coordinates |
| `top_10_hotspots.png` | Bar chart of the ten largest clusters |
| `accident_hotspots_map.html` | Interactive local Folium map |
| `kmeans_results.csv` | Same records with K-Means labels |
| `clustering_comparison.png` | Side-by-side visual comparison |

Some older output files may also exist in the folder, such as
`clustered_accidents.csv`, `dbscan_hotspots.csv`, or
`algorithm_comparison.png`. They are previous/alternate exports and are not
required by the current `main.py` workflow.

---

## 5. Fundamental machine-learning concepts

### 5.1 Unsupervised learning

Unsupervised learning works with input data but no labelled answer. In this
project, the input is the geographic position of each accident. The algorithm
tries to discover groups based on spatial similarity.

### 5.2 Latitude and longitude

- Latitude indicates north-south position.
- Longitude indicates east-west position.

They are angular geographic coordinates, not ordinary Cartesian x-y values.
That matters when calculating distances across the Earth's surface.

### 5.3 Radians

The Haversine implementation in scikit-learn expects latitude and longitude in
radians. The conversion is:

```text
radians = degrees × pi / 180
```

The original CSV remains in degrees for readability; radians are used for
distance-based processing.

### 5.4 Haversine distance

Haversine distance estimates the shortest distance between two points on a
sphere. It is appropriate for latitude and longitude because it follows the
Earth's surface more accurately than directly applying ordinary Euclidean
distance to degrees.

The project uses an approximate Earth radius of 6,371,000 metres.

### 5.5 DBSCAN

DBSCAN means **Density-Based Spatial Clustering of Applications with Noise**.
It uses two main parameters:

- `eps`: neighbourhood radius.
- `min_samples`: minimum number of records needed for a dense neighbourhood.

For this project:

```text
eps = 400 metres
min_samples = 5
```

DBSCAN classifies points conceptually as:

- **Core point:** has enough points within the epsilon neighbourhood.
- **Border point:** is close enough to a core point but may not have enough
  neighbours itself.
- **Noise point:** does not belong to a sufficiently dense region.

Noise receives the label `-1`.

### 5.6 K-distance graph

The k-distance graph sorts the distance from each point to a selected
neighbour. A sharp change, or elbow, can suggest a useful scale for `eps`.
It is a guide, not an automatic proof that a parameter is optimal.

### 5.7 K-Means

K-Means:

1. Chooses `K` centroids.
2. Assigns every record to its nearest centroid.
3. Recalculates the centroids.
4. Repeats until the assignments stabilize.

This project uses `K=5` as an illustrative comparison. It does not mean that
five is the scientifically correct number of accident regions.

### 5.8 Reproducibility

The K-Means model uses `random_state=42`. This makes its initialization
repeatable. Reproducibility also depends on using the same input data,
preprocessing, package versions, and parameter values.

### 5.9 Cluster count versus risk

A cluster containing many records means that many recorded accidents are
spatially concentrated there. It does not automatically mean:

- drivers are more likely to crash there;
- the road is more dangerous per vehicle;
- the location has the highest causal risk.

Traffic exposure and other contextual variables are needed for those claims.

---

## 6. How to run the project before presentation

Open PowerShell in:

```text
C:\Users\ycher\Desktop\ML-CEP
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

Recommended complete command:

```powershell
python main.py
```

Recommended map-only command:

```powershell
python run_map.py
```

If the browser does not open automatically, visit this address manually:

```text
http://127.0.0.1:8000/outputs/accident_hotspots_map.html
```

This generates or refreshes:

```text
outputs/hotspot_summary.csv
outputs/top_10_hotspots.png
outputs/accident_hotspots_map.html
outputs/kmeans_results.csv
outputs/clustering_comparison.png
```

Only rerun the full DBSCAN pipeline when the source data or DBSCAN settings
have changed:

```powershell
python clustering.py
```

---

## 7. How to present the project to the evaluator

### 7.1 Suggested presentation order

Use this order because it follows the logical data flow:

1. Problem statement
2. Dataset and preprocessing
3. DBSCAN method
4. Results and hotspot map
5. K-Means comparison
6. Limitations
7. Future improvements

### 7.2 Opening statement

You can begin with:

> My project is Road Accident Hotspot Detection Using Machine Learning. The
> purpose is to analyze accident coordinates and identify spatial
> concentrations of recorded accidents. I use DBSCAN as the main clustering
> algorithm because it can find dense regions without requiring a fixed number
> of clusters and can identify isolated records as noise. I also compare the
> result with K-Means and visualize the hotspots on an interactive map.

### 7.3 Explain the problem

Say:

> A raw accident dataset contains many coordinate records, but it is difficult
> to identify important geographic patterns by reading the rows directly. This
> project groups nearby accident records and presents the groups as hotspot
> regions.

Immediately state the scope carefully:

> The output represents concentrations of recorded accidents. It does not
> calculate exposure-adjusted accident risk because traffic-volume data is not
> available in the current input.

### 7.4 Explain the data

Say:

> The current prepared input contains City, Start_Lat, and Start_Lng. I
> converted the coordinate columns to numeric values, removed missing and
> invalid coordinates, and filtered an approximate Miami-area rectangle. The
> final DBSCAN result contains 3,625 valid records.

Show:

```text
outputs/city_accidents.csv
outputs/miami_cleaned.csv
```

Do not claim that this rectangle is an official Miami boundary.

### 7.5 Explain DBSCAN

Say:

> DBSCAN is a density-based clustering algorithm. It groups points when there
> are enough nearby records within a selected radius. The radius is called
> epsilon, and the minimum density requirement is called min_samples. I use
> an epsilon of 400 metres and min_samples of 5. Records that do not belong
> to a sufficiently dense region are labelled -1, which I treat as noise.

Explain why Haversine distance is used:

> Latitude and longitude are geographic coordinates, so I converted them to
> radians and used the Haversine metric, which measures distance along the
> Earth's surface.

Show:

```text
outputs/k_distance.png
outputs/k_distance_zoom.png
outputs/dbscan_parameter_sweep.csv
```

Clarify:

> The parameter sweep helps compare candidate settings. The selected
> configuration is a reasonable candidate for this project, not a proven
> universal optimum.

### 7.6 Explain the DBSCAN result

Use the verified current values:

> With the selected candidate settings, the current result contains 73
> clusters and 668 noise records out of 3,625 records. The noise percentage is
> 18.43 percent. The largest cluster contains 916 recorded accidents.

Then add:

> These numbers describe the current files and can change if the input data or
> parameters change.

Show:

```text
outputs/hotspot_summary.csv
outputs/top_10_hotspots.png
```

### 7.7 Demonstrate the interactive map

Open:

```text
outputs/accident_hotspots_map.html
```

Demonstrate these actions:

1. Point out that each cluster has a different color.
2. Point out that grey points are DBSCAN noise.
3. Click an accident marker to show its popup.
4. Toggle the accident-record layer.
5. Toggle the hotspot-center layer.
6. Explain that center markers show the average coordinate of each detected
   cluster.

Say:

> The map is generated locally using Folium and does not require a paid API
> key. Marker clustering is used so the browser does not have to display all
> records as separate visible markers at the same zoom level.

### 7.8 Explain K-Means comparison

Say:

> K-Means is a centroid-based algorithm. It requires the number of clusters in
> advance, so I use K=5 for comparison. Because K-Means uses Euclidean
> distance, I convert the geographic coordinates into a local kilometre
> approximation before fitting it. I use random_state=42 so the result is
> reproducible.

Then compare the assumptions:

| DBSCAN | K-Means |
|---|---|
| Finds density-based groups | Finds centroid-based groups |
| Does not require a fixed cluster count | Requires K in advance |
| Can label noise as `-1` | Assigns every point to a cluster |
| Can identify irregularly shaped groups | Works best with compact, centroid-shaped groups |


Show:

```text
outputs/clustering_comparison.png
outputs/kmeans_results.csv
```

Important statement:

> I do not compare the raw cluster numbers directly because cluster IDs are
> arbitrary labels. I compare the spatial patterns, assignment behaviour, and
> ability to identify noise.

### 7.9 Explain the software structure

Say:

> I separated the project into modules. `clustering.py` creates the DBSCAN
> result, `analyze_hotspots.py` calculates statistics,
> `visualize_hotspots.py` creates the map, and `compare_clustering.py`
> evaluates K-Means against DBSCAN. `main.py` combines the downstream stages
> so I can regenerate the presentation outputs without unnecessarily repeating
> the DBSCAN parameter sweep.

### 7.10 Explain limitations

Mention these openly:

1. The dataset contains accident coordinates but not traffic exposure.
2. Accident counts cannot be interpreted as per-vehicle risk.
3. The Miami study area is an approximate rectangle.
4. The current subset has no severity, temporal, weather, or road-condition
   fields.
5. DBSCAN results depend on `eps` and `min_samples`.
6. The hotspot center is a mean coordinate and may not represent an actual
   road junction.

### 7.11 Explain future improvements

Possible future work:

- Add traffic volume or vehicle-kilometres travelled.
- Add accident severity and time-of-day analysis.
- Add weather and road-condition attributes.
- Use official administrative boundaries.
- Use road-network data and identify dangerous intersections or road segments.
- Test parameter stability across multiple time periods.
- Compare additional spatial clustering methods.
- Validate hotspots against an independent dataset.

### 7.12 Closing statement

You can finish with:

> In conclusion, the project provides a complete workflow from geographic data
> cleaning to unsupervised clustering, statistical summarization, interactive
> visualization, and algorithm comparison. DBSCAN is useful here because it
> identifies dense accident concentrations and preserves isolated observations
> as noise. The result is a descriptive hotspot analysis and should be
> combined with traffic exposure and other contextual data before being used
> for safety policy decisions.

---

## 8. Practical presentation checklist

Before the evaluator arrives:

- [ ] Open the project folder in VS Code.
- [ ] Confirm that `.venv` exists.
- [ ] Run `python main.py`.
- [ ] Confirm that all five downstream outputs are refreshed.
- [ ] Open the HTML map once before the presentation.
- [ ] Keep `clustering.py` open for explaining DBSCAN.
- [ ] Keep `analyze_hotspots.py` open for explaining summary statistics.
- [ ] Keep `compare_clustering.py` open for explaining K-Means.
- [ ] Keep the output charts ready.
- [ ] Memorize the verified results: 3,625 records, 73 clusters, 668 noise
  records, 18.43% noise, and largest cluster size 916.

---

## 9. Likely evaluator questions and answers

### Why did you choose DBSCAN?

DBSCAN is suitable because it finds dense spatial regions, does not require a
predefined number of clusters, and can identify noise. Those properties are
useful for accident locations because not every record must belong to a hotspot.

### Why not use only K-Means?

K-Means requires a fixed value of K and assigns every record to a group. That
can hide isolated accidents. It is included as a comparison, while DBSCAN is
the primary method.

### What does `eps=400` mean?

It means the neighbourhood radius is approximately 400 metres after converting
the Haversine distance back from the Earth's angular representation.

### Why is `min_samples=5` used?

It requires at least five records in a local neighbourhood for a point to
qualify as a dense core. It is a selected candidate parameter, not a
universally correct value.

### Why are coordinates converted to radians?

The Haversine distance implementation expects latitude and longitude in
radians. Degrees are retained in the CSV files for readability.

### What does label `-1` mean?

It means DBSCAN classified the record as noise under the selected density
parameters. It does not mean the accident record is invalid.

### Does the largest cluster indicate the most dangerous road?

No. It indicates the largest concentration of recorded accidents in this
dataset. Exposure, traffic volume, road design, and severity would be needed
to make a stronger safety claim.

### Why do you use an approximate local projection for K-Means?

K-Means uses Euclidean distance, while latitude and longitude are geographic
coordinates. The local kilometre approximation makes the coordinate scales
more suitable for a small Miami-area region. For larger regions, a formal
projected coordinate reference system would be preferable.

### Why can’t cluster ID 2 in DBSCAN be compared with cluster ID 2 in K-Means?

Cluster IDs are assigned labels, not measurements. The number `2` has no
universal geographic meaning and can be assigned to a different group by
another algorithm or execution.

### Is this supervised or unsupervised learning?

It is unsupervised learning because there is no labelled target indicating
which records should belong to which hotspot.

### What would you improve with more time?

I would add traffic exposure, severity, time, weather, road-network, and
intersection data. I would also test parameter stability and validate the
detected hotspots against an independent source.
