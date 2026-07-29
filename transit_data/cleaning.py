"""
cleaning.py — Opportunity Distance Engine data cleaning
===========================================================
Standalone cleaning step, separated from analysis/visualization so the
team has one clear place to inspect exactly what got kept, what got
dropped, and why — before any scoring or mapping happens.

OUTPUTS:
    cleaned_stops.csv    — unified, deduplicated transit stops across all
                            agencies, with a weekday daily_trip_count
    cleaned_schools.csv  — RDU high schools only, with a school_name field
                            recovered from messy source columns, and a
                            geocode_flag marking records worth double-checking

See data_inventory_and_cleaning_log.md for the full field-by-field reasoning
behind every decision below.

SETUP:
    python3 -m venv .venv
    .venv/bin/pip install pandas geopandas shapely

RUN:
    .venv/bin/python cleaning.py
"""

import os
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# CONFIG — same paths as the main pipeline; keep these two files in sync
# ---------------------------------------------------------------------------
BASE_DIR = os.path.expanduser("~/Desktop/Google")

GTFS_FOLDERS = {
    "godurham":       os.path.join(BASE_DIR, "durham-area-transit-authority-nc-us"),
    "gocary":         os.path.join(BASE_DIR, "gocarygtfs_0"),
    "goraleigh":      os.path.join(BASE_DIR, "goraleighgtfs_july2026ext_0"),
    "gotriangle":     os.path.join(BASE_DIR, "gotrianglegodurham-nc-us"),
    "gotriangle_v2":  os.path.join(BASE_DIR, "gtfs"),
    # "UNKNOWN_1": os.path.join(BASE_DIR, "google_transit"),
    # "UNKNOWN_2": os.path.join(BASE_DIR, "google_transit (1)"),
    # "UNKNOWN_3": os.path.join(BASE_DIR, "google_transit (2)"),
}

SCHOOL_GEOJSON_PATH = os.path.join(BASE_DIR, "Public_Schools.geojson")

FILTER_TO_RDU_BBOX = True
RDU_BBOX = {"min_lon": -79.05, "max_lon": -78.30, "min_lat": 35.75, "max_lat": 36.15}

FILTER_HIGH_SCHOOLS_ONLY = True
FILTER_PUBLIC_ONLY = True  # drop non-public (private/charter) records if sch_desg present

SCHOOL_NAME_COLUMN_OVERRIDE = None  # set manually if auto-detection fails

OUTPUT_STOPS_CSV = "cleaned_stops.csv"
OUTPUT_SCHOOLS_CSV = "cleaned_schools.csv"


# ---------------------------------------------------------------------------
# STOPS: load, clean, merge across agencies
# ---------------------------------------------------------------------------
def clean_stops(path: str, agency: str) -> pd.DataFrame:
    """
    Cleaning applied:
      - subset to only the columns we actually use (drops accessibility
        codes, location_type, parent_station, etc. present in raw GTFS)
      - drop rows with missing lat/lon (can't be used in a distance calc)
      - namespace stop_id with agency prefix (raw IDs only unique per-feed)
    """
    raw = pd.read_csv(os.path.join(path, "stops.txt"))
    n_raw = len(raw)

    keep_cols = [c for c in ["stop_id", "stop_name", "stop_lat", "stop_lon"] if c in raw.columns]
    stops = raw[keep_cols].copy()

    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    n_dropped_missing_coords = n_raw - len(stops)

    stops["agency"] = agency
    stops["stop_id"] = agency + "_" + stops["stop_id"].astype(str)

    print(f"    [{agency}] stops.txt: {n_raw} raw -> {len(stops)} clean "
          f"({n_dropped_missing_coords} dropped for missing coordinates)")
    return stops


def clean_trip_frequency(path: str, agency: str) -> pd.DataFrame:
    """
    Cleaning applied:
      - filters trips to weekday-only service_ids via calendar.txt
      - explicitly flags feeds where this filter couldn't be applied
        (missing/empty calendar.txt), rather than silently using all trips
    """
    stop_times = pd.read_csv(
        os.path.join(path, "stop_times.txt"), usecols=["trip_id", "stop_id"], dtype=str
    )
    trips = pd.read_csv(os.path.join(path, "trips.txt"), usecols=["trip_id", "service_id"], dtype=str)
    n_trips_raw = trips["trip_id"].nunique()

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

    n_trips_kept = trips["trip_id"].nunique()
    print(f"    [{agency}] trips.txt: {n_trips_raw} raw trips -> {n_trips_kept} weekday trips [{note}]")

    merged = stop_times.merge(trips[["trip_id"]], on="trip_id", how="inner")
    freq = merged.groupby("stop_id").size().reset_index(name="daily_trip_count")
    freq["stop_id"] = agency + "_" + freq["stop_id"].astype(str)
    freq["freq_note"] = note
    return freq


def build_cleaned_stops_table(folders: dict) -> pd.DataFrame:
    print("Cleaning stops per agency:")
    all_stops, all_freq = [], []
    for agency, path in folders.items():
        if not os.path.isdir(path):
            print(f"    [{agency}] SKIPPED — folder not found at {path}")
            continue
        try:
            all_stops.append(clean_stops(path, agency))
            all_freq.append(clean_trip_frequency(path, agency))
        except FileNotFoundError as e:
            print(f"    [{agency}] SKIPPED — missing required GTFS file: {e}")

    stops_df = pd.concat(all_stops, ignore_index=True)
    freq_df = pd.concat(all_freq, ignore_index=True)

    merged = stops_df.merge(freq_df[["stop_id", "daily_trip_count", "freq_note"]], on="stop_id", how="left")
    n_no_trips = merged["daily_trip_count"].isna().sum()
    merged["daily_trip_count"] = merged["daily_trip_count"].fillna(0)
    if n_no_trips:
        print(f"  Note: {n_no_trips} stops had zero matched weekday trips (kept, count=0 — "
              f"could mean an unused stop, or a stop only served by frequencies.txt, which "
              f"this cleaning step doesn't currently parse)")

    # de-dup safety net: identical stop_id should be impossible after namespacing,
    # but drop exact duplicate rows if any slipped through (e.g. re-run overlap)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["stop_id"])
    if before != len(merged):
        print(f"  Dropped {before - len(merged)} duplicate stop_id rows")

    print(f"Final cleaned stops table: {len(merged)} rows\n")
    return merged


# ---------------------------------------------------------------------------
# SCHOOLS: load, clean, filter
# ---------------------------------------------------------------------------
def clean_schools(path: str) -> gpd.GeoDataFrame:
    schools = gpd.read_file(path)
    n_raw = len(schools)
    print(f"Cleaning schools: {n_raw} raw records loaded")
    print(f"  Columns available: {list(schools.columns)}")

    # --- CRS normalization ---
    schools = schools.set_crs(epsg=4326) if schools.crs is None else schools.to_crs(epsg=4326)

    # --- name-column recovery (handles shapefile 10-char truncation, e.g. school_nam) ---
    name_col = SCHOOL_NAME_COLUMN_OVERRIDE
    if not name_col:
        exact_candidates = ["school_name", "school_nam", "NAME", "Name", "SCHOOL_NAME",
                             "SchoolName", "NameLong", "NameShort", "SCH_NAME", "school"]
        name_col = next((c for c in exact_candidates if c in schools.columns), None)
    if not name_col:
        fuzzy = [c for c in schools.columns if "nam" in c.lower()]
        name_col = fuzzy[0] if fuzzy else None
        if name_col:
            print(f"  Name column not in exact list — fuzzy-matched to '{name_col}'")

    if name_col:
        schools = schools.rename(columns={name_col: "school_name"})
    else:
        print("  WARNING: no name column detected — using generic labels. "
              f"Set SCHOOL_NAME_COLUMN_OVERRIDE to one of: {list(schools.columns)}")
        schools["school_name"] = [f"School_{i}" for i in range(len(schools))]

    # --- data-quality / provenance flag ---
    # Records with a geocode correction ("ptmoved") or that failed manual
    # review are worth flagging rather than trusting silently — a single bad
    # coordinate can produce a false "transit desert" reading.
    ptmoved_col = next((c for c in ["ptmoved", "PTMOVED"] if c in schools.columns), None)
    reviewed_col = next((c for c in ["reviewed", "REVIEWED"] if c in schools.columns), None)

    if ptmoved_col or reviewed_col:
        was_moved = schools[ptmoved_col].fillna("").astype(str).str.strip().ne("") if ptmoved_col else False
        not_reviewed = schools[reviewed_col].fillna("").astype(str).str.strip().ne("1") if reviewed_col else False
        schools["geocode_flag"] = pd.Series(was_moved | not_reviewed).map(
            {True: "review_recommended", False: "ok"}
        )
        n_flagged = (schools["geocode_flag"] == "review_recommended").sum()
        print(f"  Geocode quality flag added: {n_flagged}/{len(schools)} records marked 'review_recommended'")
    else:
        schools["geocode_flag"] = "unknown"
        print("  No ptmoved/reviewed columns found — geocode_flag set to 'unknown' for all records")

    # --- type/level filters ---
    if FILTER_PUBLIC_ONLY and "sch_desg" in schools.columns:
        before = len(schools)
        schools = schools[schools["sch_desg"].astype(str).str.strip().str.lower() == "public"]
        print(f"  Filtered to public schools only: {before} -> {len(schools)}")
    elif FILTER_PUBLIC_ONLY:
        print("  FILTER_PUBLIC_ONLY is True but no 'sch_desg' column found — skipped")

    if FILTER_HIGH_SCHOOLS_ONLY and "high" in schools.columns:
        before = len(schools)
        schools = schools[schools["high"].astype(str).str.strip().str.lower() == "yes"]
        print(f"  Filtered to high schools only: {before} -> {len(schools)}")
    elif FILTER_HIGH_SCHOOLS_ONLY:
        print("  FILTER_HIGH_SCHOOLS_ONLY is True but no 'high' column found — skipped")

    if FILTER_TO_RDU_BBOX:
        before = len(schools)
        schools = schools.cx[RDU_BBOX["min_lon"]:RDU_BBOX["max_lon"], RDU_BBOX["min_lat"]:RDU_BBOX["max_lat"]]
        print(f"  Filtered to RDU bounding box: {before} -> {len(schools)}")

    # --- duplicate check (e.g. a campus digitized twice) ---
    before = len(schools)
    schools = schools.drop_duplicates(subset=["school_name", "county"] if "county" in schools.columns else ["school_name"])
    if before != len(schools):
        print(f"  Dropped {before - len(schools)} duplicate school_name+county rows")

    print(f"Final cleaned schools table: {len(schools)} rows\n")
    return schools.reset_index(drop=True)


def schools_to_flat_table(schools_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Flatten geometry into plain lat/lon columns for a clean CSV export."""
    df = schools_gdf.copy()
    df["lat"] = df.geometry.y
    df["lon"] = df.geometry.x

    keep_cols = ["school_name", "lat", "lon", "geocode_flag"]
    for optional_col in ["county", "sch_type", "sch_ptype", "sch_ctype", "lea_school"]:
        if optional_col in df.columns:
            keep_cols.append(optional_col)

    return df[keep_cols]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CLEANING TRANSIT STOPS")
    print("=" * 70)
    cleaned_stops = build_cleaned_stops_table(GTFS_FOLDERS)
    cleaned_stops.to_csv(OUTPUT_STOPS_CSV, index=False)
    print(f"Saved: {OUTPUT_STOPS_CSV}\n")

    print("=" * 70)
    print("CLEANING SCHOOL LOCATIONS")
    print("=" * 70)
    cleaned_schools_gdf = clean_schools(SCHOOL_GEOJSON_PATH)
    cleaned_schools_df = schools_to_flat_table(cleaned_schools_gdf)
    cleaned_schools_df.to_csv(OUTPUT_SCHOOLS_CSV, index=False)
    print(f"Saved: {OUTPUT_SCHOOLS_CSV}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"cleaned_stops.csv:   {len(cleaned_stops)} stops across {cleaned_stops['agency'].nunique()} agencies")
    print(f"cleaned_schools.csv: {len(cleaned_schools_df)} RDU high schools")
    if "geocode_flag" in cleaned_schools_df.columns:
        n_flagged = (cleaned_schools_df["geocode_flag"] == "review_recommended").sum()
        print(f"  -> {n_flagged} schools flagged for geocode review before trusting their scores")
    print("\nThese two files are the clean inputs for the analysis pipeline —")
    print("no further filtering/renaming should be needed downstream.")