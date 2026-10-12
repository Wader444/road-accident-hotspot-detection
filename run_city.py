"""Run the complete hotspot workflow for a city from the raw dataset."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prepare_city import DEFAULT_INPUT, prepare_city


PROJECT_DIR = Path(__file__).resolve().parent


def run_script(script: str, env: dict[str, str]) -> None:
    subprocess.run([sys.executable, script], cwd=PROJECT_DIR, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one city and run DBSCAN plus all downstream outputs."
    )
    parser.add_argument("city", help="Exact city name, case-insensitive.")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    count = prepare_city(args.city, args.input, max_rows=args.max_rows)
    env = os.environ.copy()
    env["CITY_MODE"] = "1"
    run_script("clustering.py", env)
    run_script("main.py", env)

    print(f"\nCompleted hotspot workflow for {args.city} using {count:,} records.")
    print("The generated outputs are in outputs/.")
    print("To view the map, run: python run_map.py")
    print("Note: running another city replaces the current outputs/ analysis files.")


if __name__ == "__main__":
    main()
