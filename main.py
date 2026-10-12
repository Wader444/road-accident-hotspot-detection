
import pandas as pd
from pathlib import Path

# Project paths
DATA_PATH = Path("data/US_Accidents_March23.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Change this to a city present in your dataset.
TARGET_CITY = "Miami"

# Load only the columns needed for the first version.
required_columns = ["City", "Start_Lat", "Start_Lng"]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place the CSV inside the data folder."
    )

# Read in chunks to avoid loading the full dataset into memory.
city_chunks = []
rows_collected = 0
MAX_ROWS = 10000

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=required_columns,
    chunksize=100000,
    low_memory=True
):
    # Match city names without case sensitivity or extra spaces.
    city_mask = (
        chunk["City"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        == TARGET_CITY.casefold()
    )

    city_data = chunk.loc[city_mask, required_columns]
    city_data = city_data.dropna(
        subset=["Start_Lat", "Start_Lng"]
    )

    # Discard invalid geographic coordinates.
    city_data = city_data[
        city_data["Start_Lat"].between(-90, 90)
        & city_data["Start_Lng"].between(-180, 180)
    ]

    remaining = MAX_ROWS - rows_collected

    if remaining > 0:
        selected = city_data.head(remaining)
        city_chunks.append(selected)
        rows_collected += len(selected)

    if rows_collected >= MAX_ROWS:
        break

if not city_chunks:
    raise ValueError(
        f"No valid records found for {TARGET_CITY}. "
        "Check the city name or inspect available city values."
    )

accidents = pd.concat(city_chunks, ignore_index=True)

# Remove exact duplicate coordinate records.
accidents = accidents.drop_duplicates(
    subset=["Start_Lat", "Start_Lng"]
)

accidents.to_csv(
    OUTPUT_DIR / "city_accidents.csv",
    index=False
)

print("\n--- DATASET SUMMARY ---")
print("Selected city:", TARGET_CITY)
print("Usable records:", len(accidents))
print("\nFirst five rows:")
print(accidents.head())
print("\nMissing values:")
print(accidents.isnull().sum())
print("\nSaved to outputs/city_accidents.csv")
"""Run the completed analysis, map, and clustering-comparison stages."""

from pathlib import Path

from analyze_hotspots import CHART_FILE, SUMMARY_FILE, analyze, load_results, save_chart
from compare_clustering import FIGURE_FILE, RESULTS_FILE, create_comparison, save_comparison_chart
from visualize_hotspots import OUTPUT_FILE as MAP_FILE, create_map


INPUT_FILE = Path("outputs/dbscan_results.csv")


def validate_inputs() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input is missing: {INPUT_FILE}. Run clustering.py first."
        )


def main() -> None:
    validate_inputs()
    OUTPUT_DIR = INPUT_FILE.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    summary = analyze(results)
    summary.to_csv(SUMMARY_FILE, index=False)
    save_chart(summary)
    create_map(results).save(MAP_FILE)

    compared = create_comparison(results)
    compared.to_csv(RESULTS_FILE, index=False)
    save_comparison_chart(compared)

    generated = [SUMMARY_FILE, CHART_FILE, MAP_FILE, RESULTS_FILE, FIGURE_FILE]
    print("\nProject stages completed successfully.")
    print("Generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()