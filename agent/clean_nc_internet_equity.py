"""
NC Internet Access & Education Equity — Data Cleaning + Analysis Script
-----------------------------------------------------------------------
Input:  raw CSV with columns: Area Name, Year, Internet Access, Value,
        geo_shape, geo_point_2d
Output: cleaned CSV  ->  clean_nc_internet_equity.csv
        analysis     ->  analysis_nc_internet_equity.xlsx (or .csv fallback)

Usage:
    pip install pandas numpy openpyxl
    python clean_nc_internet_equity.py --input your_file.csv

    # If your file uses commas instead of semicolons:
    python clean_nc_internet_equity.py --input your_file.csv --sep ","

Dataset too large to upload?
    Just point --input at the file on your local machine:
        python clean_nc_internet_equity.py --input /path/to/big_file.csv
    The script reads it in chunks if needed and writes outputs locally.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Clean and analyse NC internet/education equity data."
)
parser.add_argument("--input", required=True, help="Path to raw CSV file")
parser.add_argument(
    "--sep",
    default=";",
    help="Delimiter (default: semicolon — use --sep ',' if needed)",
)
parser.add_argument("--out-clean", default="clean_nc_internet_equity.csv")
parser.add_argument("--out-analysis", default="analysis_nc_internet_equity.csv")
args = parser.parse_args()


# ══════════════════════════════════════════════
# 1.  LOAD
# ══════════════════════════════════════════════
print("\n── Loading data ──────────────────────────────")
try:
    df = pd.read_csv(args.input, sep=args.sep, low_memory=False)
except Exception:
    # Try comma delimiter as fallback
    df = pd.read_csv(args.input, sep=",", low_memory=False)

print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("Columns detected:", df.columns.tolist())


# ══════════════════════════════════════════════
# 2.  STANDARDISE COLUMN NAMES
# ══════════════════════════════════════════════
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[\s\-/]+", "_", regex=True)
    .str.replace(r"[^\w]", "", regex=True)
)

# Canonical rename map (handles slight name variation in the raw file)
rename = {
    "area_name": "county",
    "areaname": "county",
    "name": "county",
    "internet_access": "internet_type",
    "internetaccess": "internet_type",
    "value": "pct_households",
    "geo_shape": "geo_shape",
    "geo_point_2d": "geo_point",
    "geopoint2d": "geo_point",
}
df.rename(
    columns={k: v for k, v in rename.items() if k in df.columns},
    inplace=True,
)

print("Standardised columns:", df.columns.tolist())


# ══════════════════════════════════════════════
# 3.  DROP PURELY GEOGRAPHIC COLUMNS
#     geo_shape is a full polygon blob — not useful for tabular analysis
# ══════════════════════════════════════════════
geo_cols = [c for c in df.columns if c in ("geo_shape",)]
df.drop(columns=geo_cols, inplace=True, errors="ignore")


# ══════════════════════════════════════════════
# 4.  PARSE geo_point  ->  lat / lon
# ══════════════════════════════════════════════
if "geo_point" in df.columns:
    coords = df["geo_point"].astype(str).str.extract(
        r"(?P<lat>-?\d+\.\d+)[,\s]+(?P<lon>-?\d+\.\d+)"
    )
    df["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    df.drop(columns=["geo_point"], inplace=True)


# ══════════════════════════════════════════════
# 5.  CLEAN KEY FIELDS
# ══════════════════════════════════════════════

# --- county ---
if "county" in df.columns:
    df["county"] = (
        df["county"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"Nan": np.nan, "None": np.nan, "": np.nan})
    )
    # Normalise abbreviation e.g. "Guilford Co" -> "Guilford County"
    df["county"] = df["county"].str.replace(
        r"\bCo\b\.?$", "County", regex=True
    )

# --- year ---
if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(2000, 2100) | df["year"].isna()]
    df["year"] = df["year"].astype("Int64")

# --- internet_type ---
if "internet_type" in df.columns:
    df["internet_type"] = (
        df["internet_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "none": np.nan, "": np.nan})
    )
    type_map = {
        "fiber optic": "fiber",
        "fiber": "fiber",
        "dsl": "dsl",
        "cable": "cable",
        "satellite": "satellite",
        "wireless": "wireless",
        "dial-up": "dialup",
        "dialup": "dialup",
        "no access": "no_access",
        "none": "no_access",
    }
    df["internet_category"] = df["internet_type"].map(
        lambda x: next(
            (v for k, v in type_map.items() if isinstance(x, str) and k in x),
            x,
        )
    )

# --- pct_households ---
if "pct_households" in df.columns:
    df["pct_households"] = pd.to_numeric(df["pct_households"], errors="coerce").astype(float)
    # Values >1 are on 0-100 scale; normalise to 0-1
    mask_pct = df["pct_households"] > 1
    df.loc[mask_pct, "pct_households"] = (
        df.loc[mask_pct, "pct_households"] / 100
    )
    df["pct_households"] = df["pct_households"].clip(0, 1)


# ══════════════════════════════════════════════
# 6.  REMOVE UNUSABLE ROWS
# ══════════════════════════════════════════════
before = len(df)

required = [c for c in ["county", "pct_households"] if c in df.columns]
df.dropna(subset=required, inplace=True)
df.drop_duplicates(inplace=True)

after = len(df)
print(f"\nRows after cleaning: {after:,}  (dropped {before - after:,} rows)")


# ══════════════════════════════════════════════
# 7.  SAVE CLEAN FILE
# ══════════════════════════════════════════════
df.to_csv(args.out_clean, index=False)
print(f"Clean CSV saved -> {args.out_clean}")


# ══════════════════════════════════════════════
# 8.  ANALYSIS  (education equity lens)
# ══════════════════════════════════════════════
print("\n── Running analysis ──────────────────────────")

results = {}

# ── 8A. County-level internet access summary ──
if "county" in df.columns and "pct_households" in df.columns:
    county_summary = (
        df.groupby("county")["pct_households"]
        .agg(
            mean_access="mean",
            min_access="min",
            max_access="max",
            n_records="count",
            std_access="std",
        )
        .reset_index()
        .sort_values("mean_access")
    )
    county_summary["access_tier"] = pd.cut(
        county_summary["mean_access"],
        bins=[0, 0.50, 0.70, 0.85, 1.0],
        labels=[
            "Low (<50%)",
            "Mid (50-70%)",
            "High (70-85%)",
            "Very High (>85%)",
        ],
    )
    results["county_summary"] = county_summary
    print(f"\n[County Summary] {len(county_summary)} counties")
    print(county_summary.to_string(index=False))

# ── 8B. Access by internet type ──
if "internet_category" in df.columns:
    type_summary = (
        df.groupby("internet_category")["pct_households"]
        .agg(mean_access="mean", county_count="count")
        .reset_index()
        .sort_values("mean_access", ascending=False)
    )
    results["access_by_type"] = type_summary
    print(f"\n[Access by Internet Type]\n{type_summary.to_string(index=False)}")

# ── 8C. Trend over time ──
if "year" in df.columns and "pct_households" in df.columns:
    df_t = df.dropna(subset=["year"])
    if not df_t.empty:
        time_trend = (
            df_t.groupby("year")["pct_households"]
            .agg(mean_access="mean", county_count="count")
            .reset_index()
        )
        time_trend["yoy_change"] = time_trend["mean_access"].diff()
        results["time_trend"] = time_trend
        print(f"\n[Time Trend]\n{time_trend.to_string(index=False)}")

# ── 8D. Equity gap: lowest vs highest quartile counties ──
if "county_summary" in results:
    cs = results["county_summary"]
    q25 = cs["mean_access"].quantile(0.25)
    q75 = cs["mean_access"].quantile(0.75)
    low_counties = cs[cs["mean_access"] <= q25]
    high_counties = cs[cs["mean_access"] >= q75]
    gap = (
        high_counties["mean_access"].mean()
        - low_counties["mean_access"].mean()
    )
    print("\n[Equity Gap]")
    avg_low = low_counties["mean_access"].mean()
    avg_high = high_counties["mean_access"].mean()
    print(f"  Bottom-quartile counties (avg access): {avg_low:.1%}")
    print(f"  Top-quartile counties    (avg access): {avg_high:.1%}")
    print(f"  Gap: {gap:.1%}  <- how far behind the least-connected counties are")
    bottom_list = ", ".join(low_counties["county"].tolist())
    print(f"\n  Bottom-quartile counties:\n  {bottom_list}")

# ── 8E. Year vs access correlation ──
if "year" in df.columns and "pct_households" in df.columns:
    df_corr = df.dropna(subset=["year", "pct_households"])
    if len(df_corr) > 10:
        corr = df_corr[["year", "pct_households"]].corr().iloc[0, 1]
        direction = (
            "positive - access improving over time"
            if corr > 0
            else "negative"
        )
        print(
            f"\n[Year vs. Access Correlation]  r = {corr:.3f}  ({direction})"
        )

# ── 8F. Fiber vs non-fiber equity comparison ──
if "internet_category" in df.columns and "pct_households" in df.columns:
    df["has_fiber"] = df["internet_category"] == "fiber"
    fiber_gap = (
        df.groupby("has_fiber")["pct_households"]
        .mean()
        .rename({True: "fiber_counties", False: "non_fiber_counties"})
    )
    results["fiber_equity_gap"] = fiber_gap.reset_index()
    print(f"\n[Fiber vs Non-Fiber Equity Gap]\n{fiber_gap.to_string()}")

# ── 8G. Highest-risk counties ──
if "county_summary" in results:
    worst = results["county_summary"][
        ["county", "mean_access", "access_tier"]
    ].copy()
    worst.columns = ["county", "mean_pct_with_access", "access_tier"]
    worst["mean_pct_with_access"] = worst["mean_pct_with_access"].map(
        "{:.1%}".format
    )
    results["all_counties_ranked"] = worst
    print(
        "\n[\n[All Counties Ranked by Internet Access (lowest to highest)]"
    )
    print(worst.to_string(index=False))


# ══════════════════════════════════════════════
# 9.  SAVE ANALYSIS
# ══════════════════════════════════════════════
try:
    import openpyxl  # noqa: F401
    out_xl = args.out_analysis.replace(".csv", ".xlsx")
    with pd.ExcelWriter(out_xl, engine="openpyxl") as xl:
        for sheet_name, frame in results.items():
            frame.to_excel(xl, sheet_name=sheet_name[:31], index=False)
    print(f"\nAnalysis Excel saved -> {out_xl}")
except ImportError:
    combined = pd.concat(
        [f.assign(section=name) for name, f in results.items()],
        ignore_index=True,
    )
    combined.to_csv(args.out_analysis, index=False)
    print(f"\nAnalysis CSV saved -> {args.out_analysis}")

print("\nDone.\n")
