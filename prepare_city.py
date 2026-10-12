"""Extract one city's accident records from the raw dataset in chunks."""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/US_Accidents_March23.csv")
OUTPUT_FILE = Path("outputs/city_accidents.csv")
REQUIRED_COLUMNS = ["City", "Start_Lat", "Start_Lng"]


def prepare_city(
    city: str,
    input_file: Path = DEFAULT_INPUT,
    output_file: Path = OUTPUT_FILE,
    max_rows: int = 10000,
) -> int:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_file}. "
            "Place US_Accidents_March23.csv in the data folder."
        )
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")

    chunks: list[pd.DataFrame] = []
    rows_collected = 0
    target = city.strip().casefold()

    for chunk in pd.read_csv(
        input_file,
        usecols=REQUIRED_COLUMNS,
        chunksize=100_000,
        low_memory=True,
    ):
        city_mask = (
            chunk["City"].fillna("").astype(str).str.strip().str.casefold() == target
        )
        selected = chunk.loc[city_mask, REQUIRED_COLUMNS].copy()
        selected["Start_Lat"] = pd.to_numeric(selected["Start_Lat"], errors="coerce")
        selected["Start_Lng"] = pd.to_numeric(selected["Start_Lng"], errors="coerce")
        selected = selected.dropna(subset=["Start_Lat", "Start_Lng"])
        selected = selected[
            selected["Start_Lat"].between(-90, 90)
            & selected["Start_Lng"].between(-180, 180)
        ]

        remaining = max_rows - rows_collected
        if remaining > 0:
            chunks.append(selected.head(remaining))
            rows_collected += min(len(selected), remaining)
        if rows_collected >= max_rows:
            break

    if not chunks or rows_collected == 0:
        raise ValueError(
            f"No valid accident records found for city '{city}'. "
            "Check the spelling or inspect city values in the raw dataset."
        )

    result = pd.concat(chunks, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one city from the raw US Accidents CSV."
    )
    parser.add_argument("city", help="Exact city name, case-insensitive.")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    count = prepare_city(args.city, args.input, max_rows=args.max_rows)
    print(f"Saved {count:,} valid records for {args.city} to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
