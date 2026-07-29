"""
clean_statewide_transit_pipeline.py
=========================================
The cleaned-data counterpart to opportunity_distance_pipeline.py, matching
its EXACT current config (all 41 NC transit agencies, all school levels,
no RDU bounding box) — but stopping BEFORE the composite transit_desert_score
is computed. Outputs raw metrics only: distance, stop density, frequency,
per school. This is what you send back for correlation testing against
education data, per the earlier decision not to define score weighting
until we know which raw metric(s) actually correlate with an outcome.

WHY A SEPARATE FILE INSTEAD OF JUST EDITING THE MAIN PIPELINE:
Keeps the two use cases cleanly separated — opportunity_distance_pipeline.py
is the full map/chart/scoring deliverable for presentations, this one is a
narrow, fast, score-free data export for statistical analysis. Editing one
config here doesn't risk breaking the other.

OUTPUT: cleaned_statewide_transit_school_metrics.csv
  One row per school:
    school_name, county, sch_type, sch_ptype, sch_ctype, geocode_flag,
    lat, lon, nearest_stop_distance_m, stops_within_radius,
    avg_daily_trips_nearby, in_service_area, n_agencies_loaded

SETUP:
    python3 -m venv .venv
    .venv/bin/pip install pandas geopandas shapely

RUN:
    .venv/bin/python clean_statewide_transit_pipeline.py
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# CONFIG — copied exactly from the current opportunity_distance_pipeline.py
# so this produces metrics for the SAME school/agency set. If you change
# BASE_DIR or the agency list in the main pipeline, mirror the change here.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transit_data")


def _resolve_prefixed_folder(parent: str, prefix: str) -> str:
    if not os.path.isdir(parent):
        return os.path.join(parent, prefix)
    matches = [d for d in os.listdir(parent) if d.startswith(prefix)]
    return os.path.join(parent, matches[0]) if matches else os.path.join(parent, prefix)


GTFS_FOLDERS = {
    "godurham":    os.path.join(BASE_DIR, "durham-area-transit-authority-nc-us"),
    "gocary":      os.path.join(BASE_DIR, "gocarygtfs_0"),
    "goraleigh":   os.path.join(BASE_DIR, "goraleighgtfs_y2026ext_0"),
    "gotriangle":  os.path.join(BASE_DIR, "gotriangle-durham-nc-us"),
    "gotriangle_v2": os.path.join(BASE_DIR, "gtfs"),
}

INCLUDE_STATEWIDE_AGENCIES = True

STATEWIDE_GTFS_FOLDERS = {
    "acta":            os.path.join(BASE_DIR, "ACTA"),
    "alamance":        os.path.join(BASE_DIR, "Alamance"),
    "apalcart":        os.path.join(BASE_DIR, "ApalCART", "gtfs"),
    "arts":            os.path.join(BASE_DIR, "ARTS"),
    "buncombe":        os.path.join(BASE_DIR, "Buncombe"),
    "carts":           os.path.join(BASE_DIR, "CARTS"),
    "cats":            os.path.join(BASE_DIR, "CATS"),
    "cherokee":        os.path.join(BASE_DIR, "Cherokee"),
    "ckrider":         os.path.join(BASE_DIR, "CKrider", "google_transit"),
    "dcts":            os.path.join(BASE_DIR, "DCTS"),
    "fast":            os.path.join(BASE_DIR, "FAST"),
    "gaston":          _resolve_prefixed_folder(os.path.join(BASE_DIR, "Gaston"), "f-gaston-area-transit"),
    "goldsboro_wayne": os.path.join(BASE_DIR, "Goldsboro-Wayne"),
    "gta":             os.path.join(BASE_DIR, "GTA", "gtfs"),
    "great":           os.path.join(BASE_DIR, "GREAT"),
    "greenway":        os.path.join(BASE_DIR, "Greenway", "UpdateGTFS_Feed_Greenway"),
    "highpoint":       os.path.join(BASE_DIR, "Highpoint"),
    "icats":           os.path.join(BASE_DIR, "ICATS"),
    "jackson_trolley": os.path.join(BASE_DIR, "JacksonTrolly"),
    "jacksonville":    os.path.join(BASE_DIR, "Jacksonville"),
    "karts":           os.path.join(BASE_DIR, "KARTS"),
    "macon_county":    os.path.join(BASE_DIR, "MaconCounty"),
    "mitchell_county": os.path.join(BASE_DIR, "MitchellCounty"),
    "mountain_project": os.path.join(BASE_DIR, "Mountain Project"),
    "part":            os.path.join(BASE_DIR, "PART", "gtfs"),
    "pats":            os.path.join(BASE_DIR, "PATS"),
    "rtp":             os.path.join(BASE_DIR, "RTP", "gtfs"),
    "rutherford":      os.path.join(BASE_DIR, "Rutherford"),
    "salisbury":       os.path.join(BASE_DIR, "Salisbury"),
    "skat":            os.path.join(BASE_DIR, "SKAT"),
    "tarriver":        os.path.join(BASE_DIR, "tarriver-nc-us"),
    "transylvania":    os.path.join(BASE_DIR, "Transylvania"),
    "wave":            os.path.join(BASE_DIR, "wave", "wave_gtfs_20221219"),
    "winston_wsta":    os.path.join(BASE_DIR, "Winston_WSTA", "gtfs"),
    "wprta":           os.path.join(BASE_DIR, "WPRTA"),
    "wta":             os.path.join(BASE_DIR, "WTA"),
}

if INCLUDE_STATEWIDE_AGENCIES:
    GTFS_FOLDERS.update(STATEWIDE_GTFS_FOLDERS)

SCHOOL_GEOJSON_PATH = os.path.join(BASE_DIR, "RTP", "Public_Schools.geojson")

# Matches the main pipeline's CURRENT settings — statewide, all levels
FILTER_TO_RDU_BBOX = False
RDU_BBOX = {"min_lon": -79.05, "max_lon": -78.30, "min_lat": 35.75, "max_lat": 36.15}
FILTER_HIGH_SCHOOLS_ONLY = False
FILTER_PUBLIC_ONLY = True
SCHOOL_NAME_COLUMN_OVERRIDE = None
SERVICE_AREA_DISTANCE_THRESHOLD_M = 5000
NEARBY_RADIUS_METERS = 800

OUTPUT_CSV = "cleaned_statewide_transit_school_metrics.csv"


# ---------------------------------------------------------------------------
# STOPS
# ---------------------------------------------------------------------------
def load_stops(path: str, agency: str) -> pd.DataFrame:
    stops = pd.read_csv(os.path.join(path, "stops.txt"))
    keep_cols = [c for c in ["stop_id", "stop_name", "stop_lat", "stop_lon"] if c in stops.columns]
    stops = stops[keep_cols].dropna(subset=["stop_lat", "stop_lon"]).copy()
    stops["agency"] = agency
    stops["stop_id"] = agency + "_" + stops["stop_id"].astype(str)
    return stops


def compute_weekday_trip_frequency(path: str, agency: str) -> pd.DataFrame:
    stop_times = pd.read_csv(os.path.join(path, "stop_times.txt"), usecols=["trip_id", "stop_id"], dtype=str)
    trips = pd.read_csv(os.path.join(path, "trips.txt"), usecols=["trip_id", "service_id"], dtype=str)

    weekday_service_ids = None
    calendar_path = os.path.join(path, "calendar.txt")
    if os.path.exists(calendar_path) and os.path.getsize(calendar_path) > 0:
        calendar = pd.read_csv(calendar_path, dtype=str)
        weekday_cols = [c for c in ["monday", "tuesday", "wednesday", "thursday", "friday"] if c in calendar.columns]
        if weekday_cols:
            is_weekday = calendar[weekday_cols].apply(lambda r: any(v == "1" for v in r), axis=1)
            weekday_service_ids = set(calendar.loc[is_weekday, "service_id"])

    if weekday_service_ids:
        trips = trips[trips["service_id"].isin(weekday_service_ids)]
        note = "weekday-filtered"
    else:
        note = "UNFILTERED (no usable calendar.txt)"

    merged = stop_times.merge(trips[["trip_id"]], on="trip_id", how="inner")
    freq = merged.groupby("stop_id").size().reset_index(name="daily_trip_count")
    freq["stop_id"] = agency + "_" + freq["stop_id"].astype(str)
    freq["freq_note"] = note
    return freq


def load_and_merge_all_agencies(folders: dict):
    all_stops, all_freq, agencies_loaded = [], [], []
    for agency, path in folders.items():
        if not os.path.isdir(path):
            print(f"  SKIP {agency}: not found -> {path}")
            continue
        try:
            all_stops.append(load_stops(path, agency))
            all_freq.append(compute_weekday_trip_frequency(path, agency))
            agencies_loaded.append(agency)
            print(f"  loaded {agency}")
        except FileNotFoundError as e:
            print(f"  SKIP {agency}: missing file -> {e}")

    if not all_stops:
        raise RuntimeError("No agencies loaded — check BASE_DIR and folder paths.")

    stops_df = pd.concat(all_stops, ignore_index=True)
    freq_df = pd.concat(all_freq, ignore_index=True)
    merged = stops_df.merge(freq_df[["stop_id", "daily_trip_count"]], on="stop_id", how="left")
    merged["daily_trip_count"] = merged["daily_trip_count"].fillna(0)
    return merged, agencies_loaded


# ---------------------------------------------------------------------------
# SCHOOLS
# ---------------------------------------------------------------------------
def load_schools(path: str) -> gpd.GeoDataFrame:
    schools = gpd.read_file(path)
    schools = schools.set_crs(epsg=4326) if schools.crs is None else schools.to_crs(epsg=4326)
    print(f"  GeoJSON columns: {list(schools.columns)}")

    name_col = SCHOOL_NAME_COLUMN_OVERRIDE
    if not name_col:
        exact = ["school_name", "school_nam", "NAME", "Name", "SCHOOL_NAME", "SchoolName", "SCH_NAME"]
        name_col = next((c for c in exact if c in schools.columns), None)
    if not name_col:
        fuzzy = [c for c in schools.columns if "nam" in c.lower()]
        name_col = fuzzy[0] if fuzzy else None
    schools = schools.rename(columns={name_col: "school_name"}) if name_col else schools.assign(
        school_name=[f"School_{i}" for i in range(len(schools))])

    ptmoved_col = "ptmoved" if "ptmoved" in schools.columns else None
    reviewed_col = "reviewed" if "reviewed" in schools.columns else None
    if ptmoved_col or reviewed_col:
        was_moved = schools[ptmoved_col].fillna("").astype(str).str.strip().ne("") if ptmoved_col else False
        not_reviewed = schools[reviewed_col].fillna("").astype(str).str.strip().ne("1") if reviewed_col else False
        schools["geocode_flag"] = pd.Series(was_moved | not_reviewed).map({True: "review_recommended", False: "ok"})
    else:
        schools["geocode_flag"] = "unknown"

    if FILTER_HIGH_SCHOOLS_ONLY and "high" in schools.columns:
        before = len(schools)
        schools = schools[schools["high"].astype(str).str.strip().str.lower() == "yes"]
        print(f"  High schools only: {before} -> {len(schools)}")

    if FILTER_PUBLIC_ONLY and "sch_desg" in schools.columns:
        before = len(schools)
        schools = schools[schools["sch_desg"].astype(str).str.strip().str.lower() == "public"]
        print(f"  Public only: {before} -> {len(schools)}")

    if FILTER_TO_RDU_BBOX:
        before = len(schools)
        schools = schools.cx[RDU_BBOX["min_lon"]:RDU_BBOX["max_lon"], RDU_BBOX["min_lat"]:RDU_BBOX["max_lat"]]
        print(f"  RDU bbox: {before} -> {len(schools)}")
    else:
        print(f"  No geographic filter — statewide, {len(schools)} schools")

    return schools.reset_index(drop=True)


# ---------------------------------------------------------------------------
# RAW METRICS — no composite score
# ---------------------------------------------------------------------------
def compute_raw_metrics(stops_df: pd.DataFrame, schools_gdf: gpd.GeoDataFrame, radius_m: int, agencies_loaded: list) -> pd.DataFrame:
    stops_gdf = gpd.GeoDataFrame(
        stops_df, geometry=[Point(xy) for xy in zip(stops_df["stop_lon"], stops_df["stop_lat"])], crs="EPSG:4326"
    )
    stops_m = stops_gdf.to_crs(epsg=32617)
    schools_m = schools_gdf.to_crs(epsg=32617).reset_index(drop=True)
    schools_m["school_row_id"] = schools_m.index

    META_COLS = {
        "county": ["county", "County", "COUNTY"],
        "sch_type": ["sch_type"], "sch_ptype": ["sch_ptype"], "sch_ctype": ["sch_ctype"],
    }
    resolved = {k: next((c for c in v if c in schools_m.columns), None) for k, v in META_COLS.items()}

    nearest = gpd.sjoin_nearest(
        schools_m[["school_row_id", "geometry"]], stops_m[["stop_id", "geometry"]], distance_col="nearest_stop_distance_m"
    ).drop_duplicates(subset="school_row_id")[["school_row_id", "nearest_stop_distance_m"]]

    schools_buffered = schools_m[["school_row_id", "geometry"]].copy()
    schools_buffered["geometry"] = schools_buffered.geometry.buffer(radius_m)
    nearby = gpd.sjoin(schools_buffered, stops_m[["daily_trip_count", "geometry"]], predicate="intersects", how="left")
    nearby_agg = nearby.groupby("school_row_id").agg(
        stops_within_radius=("daily_trip_count", "count"),
        avg_daily_trips_nearby=("daily_trip_count", "mean"),
    ).reset_index()
    nearby_agg["avg_daily_trips_nearby"] = nearby_agg["avg_daily_trips_nearby"].fillna(0).round(1)

    df = schools_m[["school_row_id", "school_name", "geocode_flag"] + [c for c in resolved.values() if c]].copy()
    df = df.rename(columns={v: k for k, v in resolved.items() if v})
    for k, v in resolved.items():
        if not v:
            df[k] = None
    df["lat"] = schools_gdf.reset_index(drop=True).geometry.y
    df["lon"] = schools_gdf.reset_index(drop=True).geometry.x
    df = df.merge(nearest, on="school_row_id", how="left").merge(nearby_agg, on="school_row_id", how="left")
    df["nearest_stop_distance_m"] = df["nearest_stop_distance_m"].round(1)
    df["stops_within_radius"] = df["stops_within_radius"].fillna(0).astype(int)
    df["avg_daily_trips_nearby"] = df["avg_daily_trips_nearby"].fillna(0)
    df["in_service_area"] = df["nearest_stop_distance_m"].notna() & (df["nearest_stop_distance_m"] <= SERVICE_AREA_DISTANCE_THRESHOLD_M)
    df["n_agencies_loaded"] = len(agencies_loaded)
    return df.drop(columns=["school_row_id"])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading GTFS feeds (all 41 configured agencies)...")
    stops_df, agencies_loaded = load_and_merge_all_agencies(GTFS_FOLDERS)
    print(f"\nTotal stops: {len(stops_df)} across {len(agencies_loaded)}/{len(GTFS_FOLDERS)} agencies loaded\n")

    print("Loading schools...")
    schools_gdf = load_schools(SCHOOL_GEOJSON_PATH)
    print()

    print("Computing raw transit metrics (NO composite score)...")
    result = compute_raw_metrics(stops_df, schools_gdf, NEARBY_RADIUS_METERS, agencies_loaded)
    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV} ({len(result)} schools)")
    print(f"  {(result['stops_within_radius']==0).mean()*100:.1f}% have zero nearby stops")
    print(f"  {(~result['in_service_area']).mean()*100:.1f}% are outside transit coverage entirely")
    print(f"  {(result['geocode_flag']=='review_recommended').mean()*100:.1f}% flagged for geocode review")
    print("\nColumns:", list(result.columns))
    print("\nThis file has NO transit_desert_score — send it back for correlation")
    print("testing against education data before any weighting decision is made.")
