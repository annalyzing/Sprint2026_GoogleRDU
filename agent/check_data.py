import os
import pandas as pd

# Load school data
df_schools = pd.read_excel("combined.xls")
df_schools_all = df_schools[df_schools["subgroup"] == "ALL"]
school_counties = (
    df_schools_all["county"].dropna().astype(str).str.strip().str.title().unique()
)

print("--- SCHOOL DATA COUNTIES (Sample) ---")
print(school_counties[:10])

csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
print("\n--- FOUND CSV FILES ---")
print(csv_files)

for fname in csv_files:
    if fname == "Custom_Report_School.csv":
        continue
    print(f"\n--- INSPECTING FILE: {fname} ---")
    try:
        df = pd.read_csv(fname, sep=";", nrows=5)
        if len(df.columns) <= 1:
            df = pd.read_csv(fname, sep=",", nrows=5)
    except Exception:
        df = pd.read_csv(fname, sep=",", nrows=5)

    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.iloc[:3, :4])