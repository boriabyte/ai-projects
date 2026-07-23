import pandas as pd
from pathlib import Path

VAL_PATH = Path("data exploration/data/rotexts/val.csv")
OUTPUT_PATH = Path("data exploration/data/rotexts/test.csv")

df = pd.read_csv(VAL_PATH)
print(f"Total val rows: {len(df)}")

# extract lines 16212 to 26212
test_df = df.iloc[16212:26212].reset_index(drop=True)

print(f"Extracted {len(test_df)} rows")

# save in the same format as cna/rocomments test.csv
test_df.to_csv(OUTPUT_PATH, index=True)

print(f"Saved to {OUTPUT_PATH}")
print(f"\nSample:")
print(test_df.head(3).to_string())