"""
time_tax_analysis.py
==========================
Tests the "Time Tax" hypothesis: does longer bus ride time (or route
miles, or transportation spending per pupil) correlate with LOWER EOG/EOC
proficiency at the district level?

GRANULARITY MISMATCH — read this first:
  NCDPI's Transportation Service Indicators data is published at the
  DISTRICT (LEA) level (one row per district: avg ride time, route miles,
  spending per pupil). NCDPI's Report Card proficiency data is published
  at the SCHOOL level. These don't join directly — this script aggregates
  school-level proficiency UP to the district level (enrollment-weighted
  mean) before testing against transportation data. This means findings
  here are about "how transportation service level in a district relates
  to that district's overall proficiency" — NOT about which individual
  school's students spend the longest on a bus, which the data doesn't
  support without additional (more granular) TIMS route-level data.

METHOD (same correlation-first approach as before — no composite score):
  1. Load + clean transportation data (one row per district)
  2. Load + clean proficiency data, aggregate to district level
     (enrollment-weighted average proficiency across a district's schools)
  3. Join on district/LEA name (fuzzy-matched — naming won't be identical)
  4. For each transportation metric (ride time, route miles, spending):
       - Pearson + Spearman correlation vs. district proficiency
       - Specific test of the "45-minute threshold" claim: split districts
         into <=45min vs >45min avg ride time, Mann-Whitney U test —
         because if the literature's claim is a THRESHOLD effect (fatigue
         kicks in past a cutoff) rather than a smooth linear one, a plain
         correlation can understate or miss it entirely; the threshold
         test checks that specific shape directly.
  5. Multivariate regression with poverty as a control (districts with
     longer routes are very likely also more rural/poorer — without
     controlling for this, a raw correlation could just be re-detecting
     the poverty-achievement gap, not a transportation effect specifically)

SETUP:
    python3 -m venv .venv
    .venv/bin/pip install pandas scipy statsmodels matplotlib rapidfuzz openpyxl

RUN:
    .venv/bin/python time_tax_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------------
# CONFIG — column names are placeholders based on what these NCDPI sources
# typically publish; confirm against your actual downloaded files and adjust.
# ---------------------------------------------------------------------------
TRANSPORT_FILE = "transportation_service_indicators.xlsx"
TRANSPORT_DISTRICT_COL = "lea_name"          # or "district", "lea", etc — check your file
TRANSPORT_RIDE_TIME_COL = "avg_ride_time_minutes"
TRANSPORT_ROUTE_MILES_COL = "avg_route_miles"      # optional — set to None if not in your file
TRANSPORT_SPENDING_COL = "transportation_spending_per_pupil"  # optional

PROFICIENCY_FILE = "school_report_cards.csv"  # or .xlsx
PROF_SCHOOL_COL = "school_name"
PROF_DISTRICT_COL = "lea_name"                # or derive from lea_code — check your file
PROF_RATE_COL = "eog_eoc_proficiency_pct"     # confirm exact column name in your download
PROF_ENROLLMENT_COL = "enrollment"            # for enrollment-weighted district average
POVERTY_COL = None  # set if your report card / combined file has a poverty column

RIDE_TIME_THRESHOLD_MIN = 45  # the literature's claimed fatigue threshold

FUZZY_MATCH_THRESHOLD = 85

OUTPUT_MATCHED_CSV = "time_tax_matched.csv"
OUTPUT_UNMATCHED_CSV = "time_tax_unmatched.csv"


# ---------------------------------------------------------------------------
# NAME NORMALIZATION (district names have their own quirks — "County
# Schools" vs "County Public Schools" vs just "County")
# ---------------------------------------------------------------------------
import re

_DISTRICT_REPLACEMENTS = {
    r"\bpublic schools\b": "schools",
    r"\bcounty schools\b": "schools",
    r"\bschool district\b": "schools",
    r"\bcity schools\b": "schools",
    r"\bschools\b": "",
    r"\bcounty\b": "",
}


def normalize_district_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    for pattern, repl in _DISTRICT_REPLACEMENTS.items():
        name = re.sub(pattern, repl, name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# 1. LOAD TRANSPORTATION DATA
# ---------------------------------------------------------------------------
def load_transportation_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    print(f"Loaded transportation file: {len(df)} district rows")

    keep_cols = [TRANSPORT_DISTRICT_COL, TRANSPORT_RIDE_TIME_COL]
    for col in [TRANSPORT_ROUTE_MILES_COL, TRANSPORT_SPENDING_COL]:
        if col and col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].copy()
    df = df.rename(columns={TRANSPORT_DISTRICT_COL: "district_name", TRANSPORT_RIDE_TIME_COL: "avg_ride_time_min"})
    df["district_norm"] = df["district_name"].apply(normalize_district_name)
    df = df.dropna(subset=["avg_ride_time_min"])
    print(f"  Cleaned: {len(df)} districts with valid ride time data\n")
    return df


# ---------------------------------------------------------------------------
# 2. LOAD + AGGREGATE PROFICIENCY DATA TO DISTRICT LEVEL
# ---------------------------------------------------------------------------
def load_and_aggregate_proficiency(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    n_schools_raw = len(df)
    print(f"Loaded proficiency file: {n_schools_raw} school rows")

    df = df.dropna(subset=[PROF_RATE_COL, PROF_DISTRICT_COL])
    df[PROF_RATE_COL] = pd.to_numeric(df[PROF_RATE_COL], errors="coerce")
    df = df.dropna(subset=[PROF_RATE_COL])

    if PROF_ENROLLMENT_COL in df.columns:
        df[PROF_ENROLLMENT_COL] = pd.to_numeric(df[PROF_ENROLLMENT_COL], errors="coerce").fillna(0)
        # enrollment-weighted average: bigger schools count more toward the
        # district average, matching how a district's overall rate is
        # actually calculated rather than a naive unweighted school average
        district_agg = df.groupby(PROF_DISTRICT_COL).apply(
            lambda g: np.average(g[PROF_RATE_COL], weights=g[PROF_ENROLLMENT_COL].replace(0, 1))
        ).reset_index(name="district_proficiency")
        print("  Aggregated using enrollment-weighted average")
    else:
        district_agg = df.groupby(PROF_DISTRICT_COL)[PROF_RATE_COL].mean().reset_index(name="district_proficiency")
        print("  WARNING: no enrollment column found — using unweighted average "
              "(a district with one tiny outlier school will be over-influenced)")

    n_schools_per_district = df.groupby(PROF_DISTRICT_COL).size().reset_index(name="n_schools")
    district_agg = district_agg.merge(n_schools_per_district, on=PROF_DISTRICT_COL)
    district_agg = district_agg.rename(columns={PROF_DISTRICT_COL: "district_name"})
    district_agg["district_norm"] = district_agg["district_name"].apply(normalize_district_name)

    if POVERTY_COL and POVERTY_COL in df.columns:
        poverty_agg = df.groupby(PROF_DISTRICT_COL)[POVERTY_COL].mean().reset_index()
        poverty_agg = poverty_agg.rename(columns={PROF_DISTRICT_COL: "district_name_temp"})
        district_agg = district_agg.merge(poverty_agg, left_on="district_name", right_on="district_name_temp", how="left")

    print(f"  Aggregated to {len(district_agg)} districts (from {n_schools_raw} schools)\n")
    return district_agg


# ---------------------------------------------------------------------------
# 3. FUZZY-MATCH districts
# ---------------------------------------------------------------------------
def match_districts(transport_df: pd.DataFrame, prof_df: pd.DataFrame, threshold: int = 85):
    matched_rows, unmatched_rows = [], []
    choices = prof_df["district_norm"].tolist()

    for _, t_row in transport_df.iterrows():
        result = process.extractOne(t_row["district_norm"], choices, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            _, score, idx = result
            merged = {**t_row.to_dict(), **prof_df.iloc[idx].to_dict(), "match_score": score}
            matched_rows.append(merged)
        else:
            unmatched_rows.append({**t_row.to_dict(), "reason": f"best score {result[1] if result else 0:.0f} below {threshold}"})

    matched_df = pd.DataFrame(matched_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)
    print(f"Matched: {len(matched_df)} districts")
    print(f"Unmatched: {len(unmatched_df)} districts (saved to {OUTPUT_UNMATCHED_CSV})")
    if len(transport_df):
        rate = round(len(matched_df) / len(transport_df) * 100, 1)
        print(f"Match rate: {rate}%\n")
    return matched_df, unmatched_df


# ---------------------------------------------------------------------------
# 4. CORRELATION + THRESHOLD TESTS
# ---------------------------------------------------------------------------
def run_correlation(matched_df: pd.DataFrame, metric_col: str, metric_label: str, out_path: str):
    data = matched_df[[metric_col, "district_proficiency"]].dropna()
    if len(data) < 10:
        print(f"  Skipping {metric_label}: only {len(data)} districts with valid data")
        return None

    r, p = stats.pearsonr(data[metric_col], data["district_proficiency"])
    rho, p_s = stats.spearmanr(data[metric_col], data["district_proficiency"])

    plt.figure(figsize=(7.5, 6))
    plt.scatter(data[metric_col], data["district_proficiency"], alpha=0.6, s=45, color="#2c7fb8")
    z = np.polyfit(data[metric_col], data["district_proficiency"], 1)
    x_line = np.linspace(data[metric_col].min(), data[metric_col].max(), 100)
    plt.plot(x_line, np.polyval(z, x_line), color="#c0392b", linestyle="--", linewidth=1.5)
    plt.xlabel(metric_label)
    plt.ylabel("District proficiency (%, enrollment-weighted)")
    plt.title(f"{metric_label} vs. district proficiency\nPearson r={r:.3f} (p={p:.3g})   Spearman ρ={rho:.3f} (p={p_s:.3g})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    sig = " *significant*" if p < 0.05 else ""
    print(f"  {metric_label}: Pearson r={r:+.3f} (p={p:.4g}){sig}   Spearman rho={rho:+.3f} (p={p_s:.4g})   n={len(data)}")
    return {"metric": metric_label, "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_s, "n": len(data)}


def test_45_minute_threshold(matched_df: pd.DataFrame, threshold_min: int, out_path: str):
    print(f"\n  Specific test: districts with avg ride time <= {threshold_min}min vs. > {threshold_min}min")
    below = matched_df[matched_df["avg_ride_time_min"] <= threshold_min]["district_proficiency"].dropna()
    above = matched_df[matched_df["avg_ride_time_min"] > threshold_min]["district_proficiency"].dropna()

    print(f"    <= {threshold_min}min: n={len(below)}, mean proficiency={below.mean():.2f}")
    print(f"    >  {threshold_min}min: n={len(above)}, mean proficiency={above.mean():.2f}")

    if len(below) >= 5 and len(above) >= 5:
        u_stat, p_val = stats.mannwhitneyu(below, above, alternative="two-sided")
        sig = " *significant*" if p_val < 0.05 else " (not significant)"
        diff = below.mean() - above.mean()
        print(f"    Mann-Whitney U p-value: {p_val:.4g}{sig}")
        print(f"    Difference (<=45min minus >45min): {diff:+.2f} percentage points")

        plt.figure(figsize=(7, 6))
        plt.boxplot([below, above], tick_labels=[f"<= {threshold_min} min", f"> {threshold_min} min"])
        plt.ylabel("District proficiency (%)")
        plt.title(f"Proficiency by {threshold_min}-minute ride time threshold\nMann-Whitney p={p_val:.3g}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"    Chart saved to {out_path}")
    else:
        print("    Not enough districts in one group to test reliably — most NC districts "
              "may not have average ride times this long; check your data's actual range.")


# ---------------------------------------------------------------------------
# 5. MULTIVARIATE REGRESSION WITH POVERTY CONTROL
# ---------------------------------------------------------------------------
def multivariate_regression(matched_df: pd.DataFrame):
    print("\n  Multivariate regression (ride time + poverty if available):")
    try:
        import statsmodels.api as sm
    except ImportError:
        print("    statsmodels not installed — skip")
        return

    predictors = ["avg_ride_time_min"]
    if POVERTY_COL and POVERTY_COL in matched_df.columns:
        predictors.append(POVERTY_COL)

    data = matched_df[predictors + ["district_proficiency"]].dropna()
    if len(data) < 15:
        print(f"    Only {len(data)} complete rows — too few for a reliable regression")
        return

    X = sm.add_constant(data[predictors])
    y = data["district_proficiency"]
    model = sm.OLS(y, X).fit()
    print(model.summary().tables[1])
    print(f"\n    R-squared: {model.rsquared:.4f} ({model.rsquared*100:.1f}% of variance explained)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Load transportation service data")
    print("=" * 70)
    transport_df = load_transportation_data(TRANSPORT_FILE)

    print("=" * 70)
    print("STEP 2: Load + aggregate proficiency data to district level")
    print("=" * 70)
    prof_df = load_and_aggregate_proficiency(PROFICIENCY_FILE)

    print("=" * 70)
    print("STEP 3: Match districts between the two datasets")
    print("=" * 70)
    matched_df, unmatched_df = match_districts(transport_df, prof_df, FUZZY_MATCH_THRESHOLD)
    matched_df.to_csv(OUTPUT_MATCHED_CSV, index=False)
    unmatched_df.to_csv(OUTPUT_UNMATCHED_CSV, index=False)

    print("\n" + "=" * 70)
    print("STEP 4: Correlation — each transportation metric vs. district proficiency")
    print("=" * 70)
    results = []
    metric_cols = {"avg_ride_time_min": "Average ride time (minutes)"}
    if TRANSPORT_ROUTE_MILES_COL and TRANSPORT_ROUTE_MILES_COL in matched_df.columns:
        metric_cols[TRANSPORT_ROUTE_MILES_COL] = "Average route miles"
    if TRANSPORT_SPENDING_COL and TRANSPORT_SPENDING_COL in matched_df.columns:
        metric_cols[TRANSPORT_SPENDING_COL] = "Transportation spending per pupil"

    for col, label in metric_cols.items():
        out_png = f"time_tax_correlation_{col}.png"
        r = run_correlation(matched_df, col, label, out_png)
        if r:
            results.append(r)

    test_45_minute_threshold(matched_df, RIDE_TIME_THRESHOLD_MIN, "time_tax_45min_threshold.png")
    multivariate_regression(matched_df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        print(pd.DataFrame(results).to_string(index=False))
    print("\nReminder: district-level correlation, not causation. Longer routes are")
    print("strongly confounded with rurality/poverty — the multivariate regression")
    print("above is the more honest number if it ran successfully.")