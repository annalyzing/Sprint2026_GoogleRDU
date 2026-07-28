import glob
import pandas as pd

# Find all txt files
txt_files = glob.glob("*.txt")
print("Found text files:", txt_files)

for fname in txt_files:
    print(f"\n--- Checking {fname} ---")
    try:
        df = pd.read_csv(fname, sep="\t", nrows=5)
        if len(df.columns) <= 1:
            df = pd.read_csv(fname, sep=",", nrows=5)
    except Exception:
        df = pd.read_csv(fname, sep=",", nrows=5)

    print("Columns found:")
    print(df.columns.tolist())
    print("\nSample rows:")
    print(df.head(2))