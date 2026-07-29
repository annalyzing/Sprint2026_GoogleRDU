"""
Opportunity Distance Engine — full local pipeline
====================================================
Run this against your local GTFS folders + school GeoJSON. Produces:
  1. cleaned/unified stops table across all transit agencies
  2. a weekday trip-frequency metric per stop
  3. per-school "transit access" metrics (nearest stop, stops within radius,
     avg frequency nearby) via a spatial join, carrying county/school-type
     metadata through for category breakdowns
  4. a plain interactive HTML map: every bus stop and every school shown as
     a solid colored dot (schools color-coded green/yellow/red by access),
     no clustering/heatmap/search — just the raw points
  5. a static ranked bar chart of "transit desert score" per school
  6. category breakdowns (by county, by school program type) as CSV + chart
  7. a score-distribution histogram and a distance-vs-frequency scatter plot
  8. a clean output CSV you can load into BigQuery later

See research_questions.md alongside this script for the questions each
output is meant to answer, for team cross-checking against the funding/
mobility/media threads.

SETUP (run once, in your project folder):
    python3 -m venv .venv
    .venv/bin/pip install pandas geopandas shapely folium matplotlib

RUN:
    .venv/bin/python opportunity_distance_pipeline.py

------------------------------------------------------------------------
CONFIG — edit this section to match your actual folder names/paths
------------------------------------------------------------------------
Based on your screenshots, your GTFS folders look like:
    Google/durham-area-transit-author.../   -> GoDurham
    Google/gocarygtfs_0/                    -> GoCary
    Google/goraleighgtfs_july2026ext_0/     -> GoRaleigh
    Google/gotrianglegodurham-nc-us/        -> GoTriangle
    Google/gtfs/                            -> possible second GoTriangle
                                                export — verify agency.txt
                                                against the one above before
                                                keeping both (avoid double-
                                                counting trips)

You also had 3 unlabeled "google_transit", "google_transit (1)",
"google_transit (2)" folders. Open agency.txt in each one FIRST — that
tells you which agency it actually is — then add/replace the entries
below with the correct label. Don't guess from folder name alone.
"""

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIG — EDIT THESE PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.expanduser("~/Desktop/Google")  # <- change if different

GTFS_FOLDERS = {
    "godurham":    os.path.join(BASE_DIR, "durham-area-transit-authority-nc-us"),
    "gocary":      os.path.join(BASE_DIR, "gocarygtfs_0"),
    "goraleigh":   os.path.join(BASE_DIR, "goraleighgtfs_july2026ext_0"),
    # gotrianglegodurham-nc-us has its GTFS files directly in the folder
    # (no nested gtfs/ subfolder) — use it as-is:
    "gotriangle":  os.path.join(BASE_DIR, "gotrianglegodurham-nc-us"),
    # There's ALSO a separate sibling "gtfs" folder in your tree — this looks
    # like a second, possibly newer/different GoTriangle export. Check its
    # agency.txt and feed_info.txt against the one above before deciding
    # whether to use this INSTEAD of "gotriangle" or as an additional feed
    # (if both are the same agency, keeping both will double-count trips):
    "gotriangle_v2": os.path.join(BASE_DIR, "gtfs"),
    # After checking agency.txt, rename/add these three:
    # "UNKNOWN_1": os.path.join(BASE_DIR, "google_transit"),
    # "UNKNOWN_2": os.path.join(BASE_DIR, "google_transit (1)"),
    # "UNKNOWN_3": os.path.join(BASE_DIR, "google_transit (2)"),
}

SCHOOL_GEOJSON_PATH = os.path.join(BASE_DIR, "Public_Schools.geojson")

# If your GeoJSON is statewide (NC DPI school locations layer), set this True
# to clip to the RDU area before doing anything else. Adjust the bbox if your
# service area is wider/narrower than this.
FILTER_TO_RDU_BBOX = True
RDU_BBOX = {  # lon/lat bounds roughly covering Wake + Durham counties
    "min_lon": -79.05,
    "max_lon": -78.30,
    "min_lat": 35.75,
    "max_lat": 36.15,
}

# If your GeoJSON includes all school levels (elementary/middle/high) and
# you only want high schools for this analysis, set True. Uses the "high"
# property == "yes", matching NC DPI's school-locations schema.
FILTER_HIGH_SCHOOLS_ONLY = True

# If you already know the exact property name for school name in your
# GeoJSON, set it here to skip auto-detection entirely. Leave None to
# auto-detect (and print candidates if detection fails).
SCHOOL_NAME_COLUMN_OVERRIDE = None  # e.g. "NameLong" or "SchoolName"

# Distance beyond which a school is considered outside RTP transit's actual
# operating area, not just "underserved" by it. This matters because the
# five loaded feeds (GoDurham, GoCary, GoRaleigh, GoTriangle) only cover the
# RTP core — counties like Franklin/Granville have no service by design, not
# by data error. 5000m (~3 miles) is a starting judgment call: comfortably
# beyond any real walk/bike-to-transit range, so a distance past this means
# "no coverage," not "weak coverage." Adjust based on team discussion.
SERVICE_AREA_DISTANCE_THRESHOLD_M = 5000

# How close counts as "nearby" for the per-school access score, in meters.
# 800m ~ a comfortable 10-minute walk; adjust as needed.
NEARBY_RADIUS_METERS = 800

OUTPUT_CSV = "school_transit_access_scores.csv"
OUTPUT_MAP_HTML = "opportunity_distance_map.html"
OUTPUT_CHART_PNG = "transit_desert_ranking.png"


# ---------------------------------------------------------------------------
# 2. IDENTIFY AGENCIES — always check agency.txt before trusting folder names
# ---------------------------------------------------------------------------
def print_agency_labels(folders: dict):
    print("=== agency.txt contents per folder (verify these before merging) ===")
    for label, path in folders.items():
        agency_file = os.path.join(path, "agency.txt")
        if os.path.exists(agency_file):
            df = pd.read_csv(agency_file)
            names = df["agency_name"].tolist() if "agency_name" in df.columns else df.iloc[:, 0].tolist()
            print(f"  {label:12s} -> {path}\n               agency_name(s): {names}")
        else:
            print(f"  {label:12s} -> {path}  [agency.txt NOT FOUND — check path]")
    print()


# ---------------------------------------------------------------------------
# 3. LOAD + CLEAN GTFS per agency
# ---------------------------------------------------------------------------
def load_stops(path: str, agency: str) -> pd.DataFrame:
    stops = pd.read_csv(os.path.join(path, "stops.txt"))
    keep_cols = [c for c in ["stop_id", "stop_name", "stop_lat", "stop_lon"] if c in stops.columns]
    stops = stops[keep_cols].dropna(subset=["stop_lat", "stop_lon"])
    stops["agency"] = agency
    # de-dupe stop_id collisions across agencies by namespacing
    stops["stop_id"] = agency + "_" + stops["stop_id"].astype(str)
    return stops


def compute_weekday_trip_frequency(path: str, agency: str) -> pd.DataFrame:
    """
    Count trips per stop that run on a typical weekday (Mon-Fri service_id).
    Falls back to a raw trip-count-per-stop if calendar.txt is missing/empty
    (common with calendar_dates-only feeds) — flagged in output.
    """
    stop_times = pd.read_csv(
        os.path.join(path, "stop_times.txt"),
        usecols=["trip_id", "stop_id"],
        dtype=str,
    )
    trips = pd.read_csv(os.path.join(path, "trips.txt"), usecols=["trip_id", "service_id"], dtype=str)

    weekday_service_ids = None
    calendar_path = os.path.join(path, "calendar.txt")
    if os.path.exists(calendar_path) and os.path.getsize(calendar_path) > 0:
        calendar = pd.read_csv(calendar_path, dtype=str)
        weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        available = [c for c in weekday_cols if c in calendar.columns]
        if available:
            is_weekday = calendar[available].apply(lambda r: any(v == "1" for v in r), axis=1)
            weekday_service_ids = set(calendar.loc[is_weekday, "service_id"])

    if weekday_service_ids:
        trips = trips[trips["service_id"].isin(weekday_service_ids)]
        note = "weekday-filtered"
    else:
        note = "UNFILTERED (no usable calendar.txt — using all trips, treat as rough estimate)"

    merged = stop_times.merge(trips[["trip_id"]], on="trip_id", how="inner")
    freq = merged.groupby("stop_id").size().reset_index(name="daily_trip_count")
    freq["stop_id"] = agency + "_" + freq["stop_id"].astype(str)
    freq["agency"] = agency
    freq["freq_note"] = note
    return freq


def load_and_merge_all_agencies(folders: dict) -> pd.DataFrame:
    all_stops, all_freq = [], []
    for agency, path in folders.items():
        if not os.path.isdir(path):
            print(f"  SKIP {agency}: path not found -> {path}")
            continue
        try:
            stops = load_stops(path, agency)
            freq = compute_weekday_trip_frequency(path, agency)
            all_stops.append(stops)
            all_freq.append(freq)
            print(f"  loaded {agency}: {len(stops)} stops ({freq['freq_note'].iloc[0] if len(freq) else 'n/a'})")
        except FileNotFoundError as e:
            print(f"  SKIP {agency}: missing GTFS file -> {e}")

    stops_df = pd.concat(all_stops, ignore_index=True)
    freq_df = pd.concat(all_freq, ignore_index=True)
    merged = stops_df.merge(freq_df[["stop_id", "daily_trip_count", "freq_note"]], on="stop_id", how="left")
    merged["daily_trip_count"] = merged["daily_trip_count"].fillna(0)
    return merged


# ---------------------------------------------------------------------------
# 4. LOAD SCHOOLS
# ---------------------------------------------------------------------------
def load_schools(path: str) -> gpd.GeoDataFrame:
    schools = gpd.read_file(path)
    if schools.crs is None:
        schools = schools.set_crs(epsg=4326)
    else:
        schools = schools.to_crs(epsg=4326)

    print(f"  GeoJSON columns found: {list(schools.columns)}")

    name_col = SCHOOL_NAME_COLUMN_OVERRIDE
    if not name_col:
        exact_candidates = ["school_name", "school_nam", "NAME", "Name", "SCHOOL_NAME", "SchoolName",
                             "NameLong", "NameShort", "LongName", "ShortName", "SCH_NAME",
                             "school", "SITE_NAME", "SiteName"]
        name_col = next((c for c in exact_candidates if c in schools.columns), None)

    if not name_col:
        # catches both full "name" and shapefile-truncated "nam" (10-char field limit)
        fuzzy = [c for c in schools.columns if "nam" in c.lower()]
        if fuzzy:
            name_col = fuzzy[0]
            print(f"  No exact name-column match — using fuzzy match: '{name_col}'")

    if name_col and name_col in schools.columns:
        schools = schools.rename(columns={name_col: "school_name"})
    else:
        print("  WARNING: could not detect a school-name column. Using generic labels.")
        print(f"  Set SCHOOL_NAME_COLUMN_OVERRIDE to one of: {list(schools.columns)}")
        schools["school_name"] = [f"School_{i}" for i in range(len(schools))]

    if FILTER_HIGH_SCHOOLS_ONLY and "high" in schools.columns:
        before = len(schools)
        schools = schools[schools["high"].astype(str).str.strip().str.lower() == "yes"]
        print(f"  Filtered to high schools only: {before} -> {len(schools)} schools")
    elif FILTER_HIGH_SCHOOLS_ONLY:
        print("  WARNING: FILTER_HIGH_SCHOOLS_ONLY is True but no 'high' column found — skipping filter")

    if FILTER_TO_RDU_BBOX:
        before = len(schools)
        schools = schools.cx[
            RDU_BBOX["min_lon"]:RDU_BBOX["max_lon"],
            RDU_BBOX["min_lat"]:RDU_BBOX["max_lat"],
        ]
        print(f"  Filtered to RDU bbox: {before} -> {len(schools)} schools")

    return schools.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. SPATIAL JOIN — per-school transit access metrics
# ---------------------------------------------------------------------------
def compute_school_access_scores(stops_df: pd.DataFrame, schools_gdf: gpd.GeoDataFrame, radius_m: int) -> pd.DataFrame:
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=[Point(xy) for xy in zip(stops_df["stop_lon"], stops_df["stop_lat"])],
        crs="EPSG:4326",
    )

    # project to a meters-based CRS for accurate distance/buffer math
    # EPSG:32617 = UTM zone 17N, correct for the RDU area
    stops_m = stops_gdf.to_crs(epsg=32617)
    schools_m = schools_gdf.to_crs(epsg=32617)

    # Extra descriptive fields carried through for category-level analysis
    # (funding/type breakdowns, county comparison) — pulled if present, else None.
    METADATA_COLS = {
        "county": ["county", "County", "COUNTY"],
        "sch_type": ["sch_type", "SCH_TYPE"],
        "sch_ptype": ["sch_ptype", "SCH_PTYPE"],
        "sch_ctype": ["sch_ctype", "SCH_CTYPE"],
    }

    def _get_meta_col(gdf, candidates):
        return next((c for c in candidates if c in gdf.columns), None)

    resolved_meta_cols = {k: _get_meta_col(schools_m, v) for k, v in METADATA_COLS.items()}

    results = []
    for idx, school in schools_m.iterrows():
        dists = stops_m.geometry.distance(school.geometry)
        nearby_mask = dists <= radius_m
        nearby_stops = stops_m[nearby_mask]

        row = {
            "school_name": school["school_name"],
            "nearest_stop_distance_m": round(dists.min(), 1) if len(dists) else None,
            "stops_within_radius": int(nearby_mask.sum()),
            "avg_daily_trips_nearby": round(nearby_stops["daily_trip_count"].mean(), 1) if len(nearby_stops) else 0,
            "lat": schools_gdf.iloc[idx].geometry.y,
            "lon": schools_gdf.iloc[idx].geometry.x,
        }
        # Distinguishes "underserved despite being in the transit network's
        # reach" from "structurally outside the network entirely" — these
        # need different policy responses (more service vs. new coverage)
        # and averaging them together hides that difference.
        row["in_service_area"] = (
            row["nearest_stop_distance_m"] is not None
            and row["nearest_stop_distance_m"] <= SERVICE_AREA_DISTANCE_THRESHOLD_M
        )
        for meta_key, source_col in resolved_meta_cols.items():
            row[meta_key] = school[source_col] if source_col else None
        results.append(row)

    df = pd.DataFrame(results)
    # higher score = worse access (more of a "transit desert")
    df["transit_desert_score"] = (
        df["nearest_stop_distance_m"].rank(ascending=False, pct=True) * 0.4
        + (1 - df["stops_within_radius"].rank(ascending=True, pct=True)) * 0.3
        + (1 - df["avg_daily_trips_nearby"].rank(ascending=True, pct=True)) * 0.3
    ).round(3)
    return df.sort_values("transit_desert_score", ascending=False)


# ---------------------------------------------------------------------------
# 6. VISUALIZATION — interactive map
# ---------------------------------------------------------------------------
def build_map(stops_df: pd.DataFrame, school_scores: pd.DataFrame, out_path: str):
    """
    Simple, plain map: every bus stop and every school is one solid dot.
    No clustering, no heatmap, no search box — just the raw points, colored
    for quick reading. Easiest version to screenshot/share for team review.
    """
    center_lat = school_scores["lat"].mean()
    center_lon = school_scores["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    # --- Bus stops: uniform solid blue dots, no size/clustering variation ---
    stop_layer = folium.FeatureGroup(name="Bus stops")
    for _, s in stops_df.iterrows():
        folium.CircleMarker(
            location=[s["stop_lat"], s["stop_lon"]],
            radius=4,
            color="#2c7fb8",
            weight=0,
            fill=True,
            fill_color="#2c7fb8",
            fill_opacity=1.0,
            popup=f"{s['stop_name']} ({s['agency']})<br>{int(s['daily_trip_count'])} trips/weekday",
        ).add_to(stop_layer)
    stop_layer.add_to(m)

    # --- Schools: solid dots, simple 3-color traffic-light scale (good/ok/bad)
    # based on terciles of the desert score — easier to read at a glance than
    # a continuous gradient.
    q1, q2 = school_scores["transit_desert_score"].quantile([1 / 3, 2 / 3])

    def score_color(score):
        if score <= q1:
            return "#2ecc71"  # good access
        elif score <= q2:
            return "#f1c40f"  # moderate
        else:
            return "#e74c3c"  # worst access

    school_layer = folium.FeatureGroup(name="High schools")
    for _, sc in school_scores.iterrows():
        folium.CircleMarker(
            location=[sc["lat"], sc["lon"]],
            radius=7,
            color="#222222",
            weight=1,
            fill=True,
            fill_color=score_color(sc["transit_desert_score"]),
            fill_opacity=1.0,
            popup=(
                f"<b>{sc['school_name']}</b><br>"
                f"County: {sc.get('county', 'n/a')}<br>"
                f"Desert score: {sc['transit_desert_score']}<br>"
                f"Nearest stop: {sc['nearest_stop_distance_m']}m<br>"
                f"Stops within {NEARBY_RADIUS_METERS}m: {sc['stops_within_radius']}<br>"
                f"Avg daily trips nearby: {sc['avg_daily_trips_nearby']}<br>"
                f"{'⚠ Outside RTP transit service area' if not sc.get('in_service_area', True) else ''}"
            ),
        ).add_to(school_layer)
    school_layer.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_path)
    print(f"Map saved to {out_path}")


def build_ranking_chart(school_scores: pd.DataFrame, out_path: str, top_n: int = 20):
    top = school_scores.head(top_n).sort_values("transit_desert_score")
    plt.figure(figsize=(9, max(4, top_n * 0.3)))
    plt.barh(top["school_name"], top["transit_desert_score"], color="#c0392b")
    plt.xlabel("Transit desert score (higher = worse access)")
    plt.title(f"Top {top_n} schools by transit access gap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")


# ---------------------------------------------------------------------------
# 7. ADDITIONAL ANALYSIS — answers specific team cross-check questions
# ---------------------------------------------------------------------------
def summarize_by_category(school_scores: pd.DataFrame, category_col: str, out_csv: str, out_png: str, min_group_size: int = 2):
    """
    Answers: 'is transit access worse for a particular county / school type?'
    (research_questions.md, items 3 and 4). Groups are dropped below
    min_group_size since a mean of 1 school isn't a meaningful comparison.
    """
    if category_col not in school_scores.columns or school_scores[category_col].isna().all():
        print(f"  Skipping category breakdown for '{category_col}': column missing/empty in source data")
        return None

    grouped = (
        school_scores.groupby(category_col)
        .agg(
            school_count=("school_name", "count"),
            avg_desert_score=("transit_desert_score", "mean"),
            avg_nearest_stop_m=("nearest_stop_distance_m", "mean"),
            pct_zero_nearby_stops=("stops_within_radius", lambda s: round((s == 0).mean() * 100, 1)),
            pct_outside_service_area=("in_service_area", lambda s: round((~s).mean() * 100, 1)),
        )
        .query("school_count >= @min_group_size")
        .sort_values("avg_desert_score", ascending=False)
        .round(3)
    )
    grouped.to_csv(out_csv)
    print(f"  {category_col} breakdown saved to {out_csv}")
    print(grouped.to_string())

    # Second view: same breakdown, but only for schools within the transit
    # network's actual reach — this is the fairer comparison if the goal is
    # "how well does RTP transit serve the schools it could plausibly serve,"
    # since out-of-area schools would otherwise drag every group's average
    # toward "bad" for a reason that has nothing to do with service quality.
    in_area = school_scores[school_scores["in_service_area"]]
    if len(in_area) < len(school_scores):
        n_excluded = len(school_scores) - len(in_area)
        print(f"  ({n_excluded} schools excluded as outside RTP's service area — "
              f"see in-service-area-only breakdown below)")
        grouped_in_area = (
            in_area.groupby(category_col)
            .agg(
                school_count=("school_name", "count"),
                avg_desert_score=("transit_desert_score", "mean"),
                avg_nearest_stop_m=("nearest_stop_distance_m", "mean"),
                pct_zero_nearby_stops=("stops_within_radius", lambda s: round((s == 0).mean() * 100, 1)),
            )
            .query("school_count >= @min_group_size")
            .sort_values("avg_desert_score", ascending=False)
            .round(3)
        )
        in_area_csv = out_csv.replace(".csv", "_in_service_area_only.csv")
        grouped_in_area.to_csv(in_area_csv)
        print(f"  In-service-area-only breakdown saved to {in_area_csv}")
        print(grouped_in_area.to_string())

    plt.figure(figsize=(8, max(3, len(grouped) * 0.5)))
    plt.barh(grouped.index.astype(str), grouped["avg_desert_score"], color="#8e44ad")
    plt.xlabel("Average transit desert score")
    plt.title(f"Average transit access gap by {category_col}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  {category_col} chart saved to {out_png}")
    return grouped


def build_distribution_chart(school_scores: pd.DataFrame, out_path: str):
    """
    Answers: 'what fraction of schools are in the worst quartile?' and shows
    whether the score distribution is continuous or clusters into distinct
    haves/have-nots groups (research_questions.md, item 2 and 7).
    """
    scores = school_scores["transit_desert_score"]
    q1, q3 = scores.quantile([0.25, 0.75])

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=20, color="#2980b9", edgecolor="white")
    plt.axvline(q3, color="#c0392b", linestyle="--", label=f"75th percentile ({q3:.2f})")
    plt.xlabel("Transit desert score")
    plt.ylabel("Number of schools")
    plt.title("Distribution of transit access scores across RDU high schools")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    worst_quartile_pct = round((scores >= q3).mean() * 100, 1)
    zero_stop_pct = round((school_scores["stops_within_radius"] == 0).mean() * 100, 1)
    print(f"  Distribution chart saved to {out_path}")
    print(f"  {worst_quartile_pct}% of schools are at/above the 75th-percentile desert score")
    print(f"  {zero_stop_pct}% of schools have ZERO transit stops within {NEARBY_RADIUS_METERS}m")


def build_distance_vs_frequency_scatter(school_scores: pd.DataFrame, out_path: str):
    """
    Answers: 'is this a no-stop-nearby problem or a stop-exists-but-barely-runs
    problem?' (research_questions.md, item 6). A school close to a stop with
    very low frequency needs a different fix (more service) than one with no
    stop at all (new route/stop) — and a school outside the transit network's
    actual service area entirely (marked with 'x') needs a different
    conversation altogether (new route/coverage decision, not a scheduling fix).
    """
    in_area = school_scores[school_scores.get("in_service_area", True) == True]
    out_area = school_scores[school_scores.get("in_service_area", True) == False]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        in_area["nearest_stop_distance_m"],
        in_area["avg_daily_trips_nearby"],
        c=in_area["transit_desert_score"],
        cmap="RdYlGn_r",
        s=60,
        edgecolor="white",
        marker="o",
        label="Within service area",
    )
    if len(out_area):
        plt.scatter(
            out_area["nearest_stop_distance_m"],
            out_area["avg_daily_trips_nearby"],
            c=out_area["transit_desert_score"],
            cmap="RdYlGn_r",
            s=90,
            marker="x",
            linewidths=2,
            label="Outside RTP service area",
        )
    plt.colorbar(sc, label="Transit desert score")
    plt.xlabel("Distance to nearest stop (m)")
    plt.ylabel("Avg daily trips at nearby stops")
    plt.title("Proximity vs. service quality — two different failure modes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Scatter chart saved to {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_agency_labels(GTFS_FOLDERS)

    print("Loading and merging GTFS feeds...")
    stops_df = load_and_merge_all_agencies(GTFS_FOLDERS)
    print(f"Total merged stops: {len(stops_df)}\n")

    print("Loading school locations...")
    schools_gdf = load_schools(SCHOOL_GEOJSON_PATH)
    print(f"Loaded {len(schools_gdf)} schools\n")

    print("Computing per-school transit access scores...")
    school_scores = compute_school_access_scores(stops_df, schools_gdf, NEARBY_RADIUS_METERS)
    school_scores.to_csv(OUTPUT_CSV, index=False)
    print(f"Scores saved to {OUTPUT_CSV}\n")
    print(school_scores.head(10).to_string(index=False))

    print("\nBuilding visualizations...")
    build_map(stops_df, school_scores, OUTPUT_MAP_HTML)
    build_ranking_chart(school_scores, OUTPUT_CHART_PNG)

    print("\nRunning additional analysis for team cross-check questions...")
    summarize_by_category(school_scores, "county", "scores_by_county.csv", "scores_by_county.png")
    summarize_by_category(school_scores, "sch_ptype", "scores_by_school_type.csv", "scores_by_school_type.png")
    build_distribution_chart(school_scores, "score_distribution.png")
    build_distance_vs_frequency_scatter(school_scores, "distance_vs_frequency.png")

    print("\nDone. Open opportunity_distance_map.html in a browser to explore interactively.")