import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))

# Load once
df = pd.read_parquet("competition_package/datasets/train.parquet")

print(df.head())
print("\nColumns:\n", df.columns)
print("\nShape:", df.shape)

# Inspect one sequence
seq0 = df[df["seq_ix"] == df["seq_ix"].iloc[0]]
print("\nOne sequence shape:", seq0.shape)
print(seq0.head(10))

# -----------------------
# Check warm-up behaviour
# -----------------------
print("\nWarm-up / prediction switch:")
print(seq0[["step_in_seq", "need_prediction"]].head(110).tail(15))

# -----------------------
# Feature statistics
# -----------------------
feature_cols = [c for c in df.columns if c.startswith(("p", "v", "dp", "dv"))]
print("\nFeature summary:")
print(df[feature_cols].describe())

# -----------------------
# Target distribution
# -----------------------
print("\nTarget summary:")
print(df[["t0", "t1"]].describe())
print("t0 |abs| > 6:", (df["t0"].abs() > 6).mean())
print("t1 |abs| > 6:", (df["t1"].abs() > 6).mean())

# -----------------------
# Plot one sequence
# -----------------------
plt.figure(figsize=(10, 4))
plt.plot(seq0["step_in_seq"], seq0["p0"], label="p0 (bid)")
plt.plot(seq0["step_in_seq"], seq0["p6"], label="p6 (ask)")
plt.axvline(99, color="red", linestyle="--", label="warm-up end")
plt.legend()
plt.title("LOB prices over one sequence")
plt.tight_layout()
plt.show()

spread = seq0["p6"] - seq0["p0"]

plt.figure(figsize=(10,4))
plt.plot(seq0["step_in_seq"], spread)
plt.axvline(99, color="red", linestyle="--")
plt.title("Bid–ask spread over sequence")
plt.xlabel("step")
plt.ylabel("spread")
plt.tight_layout()
plt.show()