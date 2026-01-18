import os
import numpy as np
import pandas as pd
import json

from pathlib import Path
from sklearn.metrics import r2_score

# Try xgboost first (fast + strong). If you don't have it installed, see install note below.
import xgboost as xgb

# Your scorer/utilities live inside competition_package
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/competition_package")
from utils import ScorerStepByStep, DataPoint


DATA_DIR = Path("competition_package/datasets")
TRAIN_FILE = DATA_DIR / "train.parquet"
VALID_SMALL_FILE = DATA_DIR / "valid_small.parquet"
VALID_FILE = DATA_DIR / "valid.parquet"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, cheap features that often help in LOB-ish data without knowing semantics.
    Assumes columns:
      - seq_ix, step_in_seq, need_prediction
      - state_* features (or similar numeric columns)
      - target columns t0, t1 (only in train/valid)
    """
    ignore = {"seq_ix", "step_in_seq", "need_prediction", "t0", "t1"}
    feat_cols = [c for c in df.columns if c not in ignore]

    X = df[feat_cols].astype(np.float32).copy()

    # --------------------
    # LOB aggregate features (cheap, high signal)
    # --------------------

    # Total depth
    bid_vol = df[[f"v{i}" for i in range(6)]].sum(axis=1)
    ask_vol = df[[f"v{i}" for i in range(6, 12)]].sum(axis=1)

    # Order book imbalance
    X["imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

    # Top-of-book spread proxy
    X["spread"] = df["p6"] - df["p0"]

    # --------------------
    # Rolling features (per sequence)
    # --------------------
    win = 10

    # IMPORTANT: imbalance/spread live in X, not df
    gX = X.groupby(df["seq_ix"])

    X["m10_imbalance"] = (
        gX["imbalance"].rolling(win).mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(np.float32)
    )

    X["m10_spread"] = (
        gX["spread"].rolling(win).mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(np.float32)
    )

    # Generic transforms
    # 1) first differences (momentum-ish)
    for c in feat_cols[:32]:  # keep it light; increase later
        X[f"d_{c}"] = df.groupby("seq_ix")[c].diff().fillna(0).astype(np.float32)

    # 2) rolling stats within each sequence (short horizon)
    win = 10
    for c in feat_cols[:16]:
        g = df.groupby("seq_ix")[c]
        X[f"m{win}_{c}"] = g.rolling(win).mean().reset_index(level=0, drop=True).fillna(0).astype(np.float32)
        X[f"s{win}_{c}"] = g.rolling(win).std().reset_index(level=0, drop=True).fillna(0).astype(np.float32)

    return X


def train_one_target(X_train, y_train, w_train, X_valid, y_valid, w_valid, seed=42):
    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",  # fast on CPU
    )

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        sample_weight_eval_set=[w_valid],
        verbose=False,
    )
    return model


def main():
    # Start on valid_small (fast iteration)
    if not VALID_SMALL_FILE.exists():
        raise FileNotFoundError(f"Missing {VALID_SMALL_FILE}. Run: python make_valid_small.py")

    print("Loading train + valid_small...")
    train = pd.read_parquet(TRAIN_FILE)
    valid = pd.read_parquet(VALID_SMALL_FILE)

    # Use only rows where scoring cares (need_prediction==True)
    train = train[train["need_prediction"] == True].reset_index(drop=True)
    valid = valid[valid["need_prediction"] == True].reset_index(drop=True)

    print("Building features...")
    X_train = build_features(train)
    X_valid = build_features(valid)

    # --- Save feature schema + meta for inference (solution_xgb.py) ---
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # Base feature cols used before adding d_/m10_/s10_
    ignore = {"seq_ix", "step_in_seq", "need_prediction", "t0", "t1"}
    feat_cols = [c for c in train.columns if c not in ignore]

    meta = {
        "feat_cols": feat_cols,
        "diff_cols": feat_cols[:32],
        "roll_cols": feat_cols[:16],
        "roll_window": 10,
        "feature_names": list(X_train.columns),
    }

    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump(meta, f)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(list(X_train.columns), f)

    print(f"Saved feature meta + names to {out_dir}/feature_meta.json and {out_dir}/feature_names.json")

    y0_train = train["t0"].astype(np.float32).values
    y1_train = train["t1"].astype(np.float32).values
    y0_valid = valid["t0"].astype(np.float32).values
    y1_valid = valid["t1"].astype(np.float32).values

    # Sample weights: proportional to |target| (matches weighted Pearson vibe)
    w0_train = np.abs(y0_train) + 1e-3
    w1_train = np.abs(y1_train) + 1e-3
    w0_valid = np.abs(y0_valid) + 1e-3
    w1_valid = np.abs(y1_valid) + 1e-3

    print("Training XGB for t0...")
    m0 = train_one_target(X_train, y0_train, w0_train, X_valid, y0_valid, w0_valid)

    print("Training XGB for t1...")
    m1 = train_one_target(X_train, y1_train, w1_train, X_valid, y1_valid, w1_valid)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    m0.save_model(out_dir / "xgb_t0.json")
    m1.save_model(out_dir / "xgb_t1.json")
    print(f"Saved models to {out_dir}/xgb_t0.json and {out_dir}/xgb_t1.json")

    # Quick sanity metric (not the official one)
    p0 = m0.predict(X_valid)
    p1 = m1.predict(X_valid)
    print("Quick check (R^2 on valid_small, unweighted):")
    print(" t0:", r2_score(y0_valid, p0))
    print(" t1:", r2_score(y1_valid, p1))

    print("\nNext: wire these models into a solution.py predictor (and score with ScorerStepByStep).")


if __name__ == "__main__":
    main()