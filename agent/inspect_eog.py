import pandas as pd

df = pd.read_csv("Disag_2024-25_Data.txt", sep="\t", low_memory=False)
df.columns = df.columns.str.strip().str.lower()

print("--- Data Snapshot ---")
print("Unique Types:", df['type'].unique() if 'type' in df.columns else "N/A")
print("Unique Subjects:", df['subject'].unique() if 'subject' in df.columns else "N/A")
print("Unique Grades:", df['grade'].unique() if 'grade' in df.columns else "N/A")

print("\n--- Sample Name Entries ---")
print(df['name'].value_counts().head(15))