import pandas as pd

FILE = "data exploration/data/big csv/VAL_all_annotated.csv"

df = pd.read_csv(FILE)
df = df.iloc[:25395]  # keep rows 0 to 347522 (exclusive of 347523)
df.to_csv(FILE, index=False)

print(f"Done. Kept {len(df)} rows.")