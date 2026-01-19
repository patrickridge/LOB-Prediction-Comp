# make_valid_tiny.py
import pandas as pd
from pathlib import Path

VALID = Path("competition_package/datasets/valid.parquet")
OUT = Path("competition_package/datasets/valid_tiny.parquet")

N_SEQS = 100

df = pd.read_parquet(VALID)
seqs = df["seq_ix"].drop_duplicates().iloc[:N_SEQS].tolist()
tiny = df[df["seq_ix"].isin(seqs)].copy()

tiny.to_parquet(OUT, index=False)
print("Wrote:", OUT, "rows:", len(tiny), "seqs:", len(seqs))