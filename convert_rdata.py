import pyreadr

result = pyreadr.read_r(
    "data/raw/labs/lab_reports_data.rdata"
)

print(result.keys())

df = list(result.values())[0]

print(df.head())

df.to_csv(
    "data/raw/labs/lab_reports_data.csv",
    index=False
)

print("CSV created successfully!")