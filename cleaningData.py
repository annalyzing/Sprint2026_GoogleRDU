import pandas as pd

locations = pd.read_excel('rcd_location.xlsx', dtype=str)
attendance = pd.read_excel('rcd_acc_spg2.xlsx', dtype=str)

combined = pd.merge(
    attendance,
    locations,
    on=['agency_code', 'year'],   # match on school AND year, not just school
    how='left'                     # keep every attendance row, even if a location match is missing
)


group2025 = combined[combined['year'] == '2025']

# all2025 = group2025[group2025['subgroup'] == 'ALL']


# print(combined.shape)
# print(group2025.head())

# print(group2025.columns.values)


group2025.to_csv("all2025.csv", index=False)
