"""Create an interactive Folium map from the existing DBSCAN results."""

from colorsys import hsv_to_rgb
from html import escape
from pathlib import Path

import folium
import pandas as pd
from branca.element import Element
from folium.plugins import MarkerCluster


INPUT_FILE = Path("outputs/dbscan_results.csv")
CENTERS_FILE = Path("outputs/dbscan_hotspot_centers.csv")
OUTPUT_FILE = Path("outputs/accident_hotspots_map.html")
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
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["DBSCAN_Cluster"] = df["DBSCAN_Cluster"].astype(int)
    if df.empty:
        raise ValueError(f"{path} contains no valid accident coordinates.")
    return df


def popup_html(row: pd.Series) -> str:
    fields = [
        ("Cluster", int(row["DBSCAN_Cluster"])),
        ("Latitude", f"{row['Start_Lat']:.6f}"),
        ("Longitude", f"{row['Start_Lng']:.6f}"),
    ]
    for column, value in row.items():
        if column in {"Start_Lat", "Start_Lng", "DBSCAN_Cluster"}:
            continue
        if pd.notna(value):
            fields.append((column, value))
    rows = "".join(
        f"<tr><th style='text-align:left;padding-right:8px'>{escape(str(key))}</th>"
        f"<td>{escape(str(value))}</td></tr>"
        for key, value in fields
    )
    return f"<table>{rows}</table>"


def add_legend(map_object: folium.Map, colors: dict[int, str]) -> None:
    entries = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;"
        f"background:{color};margin-right:5px'></span>Cluster {cluster}</div>"
        for cluster, color in sorted(colors.items())
    )
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; border: 2px solid #777; padding: 10px;
                max-height: 280px; overflow-y: auto; font-size: 12px;">
      <b>DBSCAN clusters</b>{entries}
      <div><span style="display:inline-block;width:12px;height:12px;
                 background:#808080;margin-right:5px"></span>Noise (-1)</div>
    </div>
    """
    map_object.get_root().html.add_child(Element(legend))


def create_map(df: pd.DataFrame) -> folium.Map:
    center = [df["Start_Lat"].mean(), df["Start_Lng"].mean()]
    map_object = folium.Map(location=center, zoom_start=11, control_scale=True)

    clusters = sorted(cluster for cluster in df["DBSCAN_Cluster"].unique() if cluster != -1)
    colors = {
        cluster: "#{:02x}{:02x}{:02x}".format(
            *(round(channel * 255) for channel in hsv_to_rgb(index / max(len(clusters), 1), 0.75, 0.85))
        )
        for index, cluster in enumerate(clusters)
    }

    accidents_layer = folium.FeatureGroup(name="Accident records", show=True)
    marker_cluster = MarkerCluster(name="Grouped accident markers").add_to(accidents_layer)
    for _, row in df.iterrows():
        cluster = int(row["DBSCAN_Cluster"])
        color = "#808080" if cluster == -1 else colors[cluster]
        folium.CircleMarker(
            location=[row["Start_Lat"], row["Start_Lng"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(popup_html(row), max_width=350),
        ).add_to(marker_cluster)
    accidents_layer.add_to(map_object)

    if CENTERS_FILE.exists():
        centers = pd.read_csv(CENTERS_FILE)
        center_columns = {"DBSCAN_Cluster", "center_lat", "center_lng", "accident_records"}
        if center_columns.issubset(centers.columns):
            centers_layer = folium.FeatureGroup(name="Hotspot centers", show=True)
            for _, row in centers.iterrows():
                cluster = int(row["DBSCAN_Cluster"])
                color = colors.get(cluster, "#808080")
                folium.Marker(
                    [row["center_lat"], row["center_lng"]],
                    tooltip=f"Cluster {cluster}: {int(row['accident_records'])} records",
                    icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                ).add_to(centers_layer)
            centers_layer.add_to(map_object)

    folium.LayerControl().add_to(map_object)
    add_legend(map_object, colors)
    return map_object


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    accident_data = load_results()
    create_map(accident_data).save(OUTPUT_FILE)
    print(f"Saved interactive map for {len(accident_data):,} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
