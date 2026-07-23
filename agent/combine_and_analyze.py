import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

print("--- Step 1: Loading & Filtering NC DPI EOG Data ---")
eog_file = "Disag_2024-25_Data.txt"

# Load tab-delimited file
df_eog = pd.read_csv(eog_file, sep="\t", low_memory=False)
df_eog.columns = (
    df_eog.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
)

# Filter strictly for composite totals
if "subgroup" in df_eog.columns:
    df_eog = df_eog[df_eog["subgroup"].astype(str).str.upper() == "ALL"]

if "subject" in df_eog.columns:
    df_eog = df_eog[df_eog["subject"].astype(str).str.upper() == "ALL"]

if "grade" in df_eog.columns:
    df_eog = df_eog[df_eog["grade"].astype(str).str.upper() == "ALL"]

if "type" in df_eog.columns:
    df_eog = df_eog[df_eog["type"].astype(str).str.upper() == "ALL"]

# Exclude State & Regional aggregate rows
exclude_patterns = r"State of North Carolina|SBE Region|Charter|State Operated"
df_eog = df_eog[
    ~df_eog["name"].astype(str).str.contains(exclude_patterns, case=False, na=False)
].copy()

# Extract student counts and proficiency values
df_eog["num_tested"] = pd.to_numeric(
    df_eog["num_tested"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
)

if "pct_glp" in df_eog.columns:
    df_eog["pass_pct"] = pd.to_numeric(df_eog["pct_glp"], errors="coerce")
elif "pct_notprof" in df_eog.columns:
    df_eog["pass_pct"] = 100.0 - pd.to_numeric(df_eog["pct_notprof"], errors="coerce")
else:
    for col in ["pct_l3", "pct_l4", "pct_l5"]:
        if col in df_eog.columns:
            df_eog[col] = pd.to_numeric(df_eog[col], errors="coerce")
    df_eog["pass_pct"] = df_eog[
        [c for c in ["pct_l3", "pct_l4", "pct_l5"] if c in df_eog.columns]
    ].sum(axis=1)

# Clean District Name to County standard
df_eog["county_clean"] = (
    df_eog["name"]
    .astype(str)
    .str.replace(r"\bCounty\b|\bSchools\b|\bCity\b|\bPublic\b", "", regex=True)
    .str.strip()
    .str.title()
)

# Weighted county proficiency average
def weighted_avg(group):
    valid = group.dropna(subset=["pass_pct", "num_tested"])
    if len(valid) == 0 or valid["num_tested"].sum() == 0:
        return group["pass_pct"].mean()
    return (valid["pass_pct"] * valid["num_tested"]).sum() / valid["num_tested"].sum()

county_eog = (
    df_eog.groupby("county_clean")
    .apply(weighted_avg)
    .reset_index(name="eog_pass_rate")
    .dropna()
)

print("\n--- Step 2: Loading Internet Equity Data ---")
if os.path.exists("raw_internet_data.csv.csv"):
    df_internet = pd.read_csv("raw_internet_data.csv.csv", sep=";", low_memory=False)
    if len(df_internet.columns) <= 1:
        df_internet = pd.read_csv("raw_internet_data.csv.csv", sep=",", low_memory=False)
else:
    df_internet = pd.read_csv("internet-access.csv", sep=";", low_memory=False)

df_internet.columns = (
    df_internet.columns.str.strip().str.lower().str.replace(" ", "_")
)

i_county_col = "county" if "county" in df_internet.columns else "area_name"
i_val_col = "pct_households" if "pct_households" in df_internet.columns else "value"

df_internet["county_clean"] = (
    df_internet[i_county_col]
    .astype(str)
    .str.replace(r"\bCounty\b", "", regex=True)
    .str.strip()
    .str.title()
)

df_internet["internet_access_pct"] = pd.to_numeric(
    df_internet[i_val_col], errors="coerce"
).astype(float)

mask = df_internet["internet_access_pct"] > 1
df_internet.loc[mask, "internet_access_pct"] /= 100.0

county_internet = (
    df_internet.groupby("county_clean")["internet_access_pct"]
    .mean()
    .reset_index()
)
county_internet["internet_pct"] = county_internet["internet_access_pct"] * 100

print("\n--- Step 3: Merging & Calculating Pearson Correlation ---")
merged_df = pd.merge(
    county_eog, county_internet, on="county_clean", how="inner"
).dropna()

r_val, p_val = stats.pearsonr(merged_df["internet_pct"], merged_df["eog_pass_rate"])

print("\n=======================================================")
print("             EOG vs INTERNET CORRELATION RESULTS       ")
print("=======================================================")
print(f"Counties Analyzed:       {len(merged_df)}")
print(f"Pearson Correlation (r): {r_val:.4f}")
print(f"p-value:                 {p_val:.4e}")
print("=======================================================\n")

# ---------------------------------------------------------
print("--- Step 4: Generating Interactive Line Graph ---")

# Sort counties alphabetically or by EOG performance for line trajectory
merged_df = merged_df.sort_values(by="eog_pass_rate", ascending=True).reset_index(
    drop=True
)

fig = go.Figure()

# EOG Proficiency Line
fig.add_trace(
    go.Scatter(
        x=merged_df["county_clean"],
        y=merged_df["eog_pass_rate"],
        mode="lines+markers",
        name="EOG Grade Level Proficiency (%)",
        line=dict(color="#2b8a3e", width=3),
        marker=dict(size=6),
        hovertemplate="<b>%{x} County</b><br>EOG Proficiency: %{y:.1f}%<extra></extra>",
    )
)

# Household Internet Access Line
fig.add_trace(
    go.Scatter(
        x=merged_df["county_clean"],
        y=merged_df["internet_pct"],
        mode="lines+markers",
        name="Household Internet Access (%)",
        line=dict(color="#2b8cbe", width=3, dash="dash"),
        marker=dict(size=6),
        hovertemplate="<b>%{x} County</b><br>Internet Access: %{y:.1f}%<extra></extra>",
    )
)

fig.update_layout(
    title=dict(
        text=f"NC Counties: Household Internet Access vs. State EOG Proficiency<br><sup>Pearson Correlation (r) = {r_val:.3f} | p-value = {p_val:.4f} ({len(merged_df)} Counties)</sup>",
        font=dict(size=18),
    ),
    xaxis=dict(
        title="NC Counties (Ordered by Increasing EOG Proficiency)",
        tickangle=-45,
        dtick=1,
    ),
    yaxis=dict(title="Percentage (%)", range=[0, 100]),
    height=650,
    margin=dict(l=50, r=50, t=100, b=150),
    legend=dict(x=0.01, y=0.98, orientation="h", bgcolor="rgba(255,255,255,0.8)"),
    template="plotly_white",
)

output_html = "eog_vs_internet_line_graph.html"
fig.write_html(output_html)
print(f"Line graph saved -> {output_html}")

import webbrowser

webbrowser.open(os.path.abspath(output_html))