import pandas as pd

locations = pd.read_excel('data-files\\raw_student-data\\rcd_location.xlsx', dtype=str)
# attendance = pd.read_excel('data-files\\raw_student-data\\rcd_acc_cgr.xlsx', dtype=str)

# combined = pd.merge(
#     attendance,
#     locations,
#     on=['agency_code', 'year'],   # match on school AND year, not just school
#     how='left'                     # keep every attendance row, even if a location match is missing
# )


# group2025 = combined[combined['year'] == '2025']

# all2025 = group2025[group2025['subgroup'] == 'ALL']


# # print(combined.shape)
print(locations.head())

# print(all2025.columns.values)

# all2025.to_csv("2025gradrates.csv", index=False)


