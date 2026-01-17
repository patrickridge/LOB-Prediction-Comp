# make_valid_small.py (run from competition_package)
import pandas as pd

df = pd.read_parquet("competition_package/datasets/valid.parquet")
seqs = df["seq_ix"].unique()[:100]   # try 50 or 100
df_small = df[df["seq_ix"].isin(seqs)]
df_small.to_parquet("competition_package/datasets/valid_small.parquet", index=False)

print("Saved:", df_small.shape)

# test_file = f"{CURRENT_DIR}/../datasets/valid_small.parquet"

print("Loading valid.parquet...")
print("Selecting sequences:", len(seqs))
print("Saving to:", "competition_package/datasets/valid_small.parquet")
print("Done. Rows:", len(df_small))