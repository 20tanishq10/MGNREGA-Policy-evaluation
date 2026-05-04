from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
DATA_PATH = ROOT / "final_analysis_dataset.csv"
SHP_PATH = ROOT / "shapefile" / "DISTRICT_BOUNDARY.shp"
MAP_DIR = ROOT / "final_maps"
MAP_DIR.mkdir(exist_ok=True)

PLOT_COLS = ["composite_index", "shock", "distribution", "income", "distortion", "structural"]


def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", "", x)
    return x


def extract_district_from_region(region_id: str) -> str:
    if pd.isna(region_id):
        return ""
    text = str(region_id)
    district = text.split("_")[0]
    return district


def find_district_col(gdf: gpd.GeoDataFrame) -> str:
    candidate_cols = [c for c in gdf.columns if any(k in c.lower() for k in ["district", "dist", "dtname", "name"])]
    for c in candidate_cols:
        if gdf[c].dtype == object:
            return c
    raise ValueError(
        "District name column not found in shapefile attributes. "
        "Please provide a complete shapefile with .dbf containing district names."
    )


def apply_manual_mapping(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    # Add known mismatch corrections here if needed.
    manual_map = {
        "bangaloreurban": "bengaluruurban",
        "bangalorerural": "bengalururural",
        "puducherry": "pondicherry",
    }
    df[key_col] = df[key_col].replace(manual_map)
    return df


def make_map(gdf: gpd.GeoDataFrame, col: str, title: str, out_file: Path, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(
        column=col,
        cmap=cmap,
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close(fig)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    if not SHP_PATH.exists():
        raise FileNotFoundError(f"Missing shapefile: {SHP_PATH}")

    # Guard: a complete shapefile must include DBF for district attributes.
    dbf_path = SHP_PATH.with_suffix(".dbf")
    if not dbf_path.exists():
        raise FileNotFoundError(
            f"Missing shapefile attribute file: {dbf_path}. "
            "Current shapefile is incomplete (.shp only), cannot merge by district name."
        )

    data = pd.read_csv(DATA_PATH)
    shp = gpd.read_file(SHP_PATH)

    data["district"] = data["region_id"].map(extract_district_from_region)
    data["district_key"] = data["district"].map(normalize_text)
    data = apply_manual_mapping(data, "district_key")

    district_col = find_district_col(shp)
    shp["district_key"] = shp[district_col].map(normalize_text)
    shp = apply_manual_mapping(shp, "district_key")

    latest = data[data["year"] == 2024].copy()
    pre = data[data["year"].between(2015, 2019)].groupby("district_key", as_index=False)[PLOT_COLS].mean()
    post = data[data["year"].between(2020, 2024)].groupby("district_key", as_index=False)[PLOT_COLS].mean()
    latest_agg = latest.groupby("district_key", as_index=False)[PLOT_COLS].mean()

    # Required map: 2024 composite
    map_2024 = shp.merge(latest_agg[["district_key", "composite_index"]], on="district_key", how="left")
    make_map(map_2024, "composite_index", "MGNREGA Composite Index (2024)", MAP_DIR / "india_composite_map_2024.png", cmap="RdYlGn")

    # Required pillar maps: using post-COVID average for stable comparisons
    for col, fname in [
        ("shock", "india_shock_map.png"),
        ("distribution", "india_distribution_map.png"),
        ("income", "india_income_map.png"),
        ("distortion", "india_distortion_map.png"),
        ("structural", "india_structural_map.png"),
    ]:
        m = shp.merge(post[["district_key", col]], on="district_key", how="left")
        make_map(m, col, f"{col.title()} Score (Post-COVID Avg 2020-2024)", MAP_DIR / fname, cmap="viridis")

    # Optional percentile maps for advanced contrast
    percentile = latest_agg[["district_key", "composite_index"]].copy()
    percentile["q"] = pd.qcut(percentile["composite_index"], q=5, labels=False, duplicates="drop")
    pm = shp.merge(percentile, on="district_key", how="left")
    make_map(pm, "q", "Composite Index Percentile Bins (2024)", MAP_DIR / "india_composite_percentile_map_2024.png", cmap="YlGnBu")

    print(f"Maps generated in: {MAP_DIR.resolve()}")


if __name__ == "__main__":
    main()
