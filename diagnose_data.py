
import pandas as pd
import numpy as np
from pathlib import Path

path = Path("outputs/city_accidents.csv")

if not path.exists():
    raise FileNotFoundError(f"Cannot find {path.resolve()}")

df = pd.read_csv(path)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- AVAILABLE COLUMNS ---")
print(df.columns.tolist())

required = {"Start_Lat", "Start_Lng"}
if not required.issubset(df.columns):
    raise ValueError("Start_Lat or Start_Lng is missing.")

df["Start_Lat"] = pd.to_numeric(df["Start_Lat"], errors="coerce")
df["Start_Lng"] = pd.to_numeric(df["Start_Lng"], errors="coerce")

df = df.dropna(subset=["Start_Lat", "Start_Lng"])

print("\n--- COORDINATE RANGES ---")
print(df[["Start_Lat", "Start_Lng"]].agg(["min", "max"]))

print("\n--- COORDINATE SUMMARY ---")
print(df[["Start_Lat", "Start_Lng"]].describe())

if "City" in df.columns:
    print("\n--- TOP CITY COUNTS ---")
    print(df["City"].value_counts(dropna=False).head(15))

    print("\n--- RECORDS OUTSIDE TARGET CITY ---")
    print(
        df.loc[
            df["City"].fillna("").str.casefold() != "miami",
            ["City", "Start_Lat", "Start_Lng"]
        ].head(15).to_string(index=False)
    )

print("\n--- GEOGRAPHIC OUTLIERS ---")
# Miami-area coordinates are used only as a diagnostic here.
# Replace these bounds if your intended city is different.
outside = ~(
    df["Start_Lat"].between(25.0, 26.5)
    & df["Start_Lng"].between(-81.0, -79.5)
)

print("Points outside diagnostic bounds:", int(outside.sum()))
print(
    df.loc[outside, ["Start_Lat", "Start_Lng"]]
    .head(15)
    .to_string(index=False)
)

print("\n--- PHASE 1 DATA DIAGNOSTICS COMPLETE ---")