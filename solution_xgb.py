import os
import sys
import json
from collections import deque
import numpy as np

# Adjust path to import utils from competition_package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "competition_package"))

from utils import DataPoint, ScorerStepByStep
import xgboost as xgb


class PredictionModel:
    """
    XGBoost solution matching train_xgb.py feature engineering:
      - base feat cols (from feature_meta.json)
      - d_{c} for first N diff cols within each seq
      - rolling mean/std (win=10) for first M roll cols within each seq
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load boosters
        self.model_t0 = xgb.Booster()
        self.model_t1 = xgb.Booster()
        self.model_t0.load_model(os.path.join(base_dir, "artifacts", "xgb_t0.json"))
        self.model_t1.load_model(os.path.join(base_dir, "artifacts", "xgb_t1.json"))

        # Load feature meta written by train_xgb.py
        meta_path = os.path.join(base_dir, "artifacts", "feature_meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.feat_cols = self.meta["feat_cols"]
        self.diff_cols = self.meta["diff_cols"]
        self.roll_cols = self.meta["roll_cols"]
        self.win = int(self.meta["roll_window"])
        self.feature_names = self.meta["feature_names"]

        # Per-sequence state
        self.current_seq_ix = None
        self.prev_vals = {}  # for diffs
        self.roll_bufs = {c: deque(maxlen=self.win) for c in self.roll_cols}
        # buffers for engineered rolling features
        self.roll_bufs["imbalance"] = deque(maxlen=self.win)
        self.roll_bufs["spread"] = deque(maxlen=self.win)

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.prev_vals = {}
        self.roll_bufs = {c: deque(maxlen=self.win) for c in self.roll_cols}
        self.roll_bufs["imbalance"] = deque(maxlen=self.win)
        self.roll_bufs["spread"] = deque(maxlen=self.win)

    def _state_to_base_features(self, data_point: DataPoint) -> dict:
        """
        Build a dict {feat_col: value} for the base columns used in training.
        Assumption: DataPoint.state corresponds (in order) to feat_cols.
        If your competition utils map differently, we’ll adapt.
        """
        s = data_point.state.astype(np.float32, copy=False)

        if len(s) != len(self.feat_cols):
            raise ValueError(
                f"DataPoint.state length {len(s)} does not match feat_cols length {len(self.feat_cols)}. "
                "If state includes extra fields, we need to map columns explicitly."
            )

        return {c: float(s[i]) for i, c in enumerate(self.feat_cols)}

    # --- FIX 1: moved out of _state_to_base_features and made a real class staticmethod ---
    @staticmethod
    def _lob_derived(base: dict) -> tuple[float, float]:
        # bid v0..v5, ask v6..v11
        bid_vol = sum(base[f"v{i}"] for i in range(6))
        ask_vol = sum(base[f"v{i}"] for i in range(6, 12))
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

        # spread proxy (assumes p0 best bid, p6 best ask as you used in training)
        spread = base["p6"] - base["p0"]
        return float(imbalance), float(spread)

    @staticmethod
    def _mean_std(buf: deque):
        if len(buf) == 0:
            return 0.0, 0.0
        arr = np.asarray(buf, dtype=np.float32)
        return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # reset per sequence
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_seq(data_point.seq_ix)

        base = self._state_to_base_features(data_point)

        imb, spr = self._lob_derived(base)
        base["imbalance"] = imb
        base["spread"] = spr

        # keep engineered rolling buffers updated
        self.roll_bufs["imbalance"].append(imb)
        self.roll_bufs["spread"].append(spr)

        # update rolling buffers (must happen every step, not just when need_prediction)
        for c in self.roll_cols:
            self.roll_bufs[c].append(base[c])

        if not data_point.need_prediction:
            # still update prev_vals for diffs
            for c in self.diff_cols:
                self.prev_vals[c] = base[c]
            return None

        # Build full feature row in the EXACT column order used in training
        row = {}

        # base features
        for c in self.feat_cols:
            row[c] = base[c]

        # --- FIX 2: add engineered base features AFTER row exists ---
        row["imbalance"] = base["imbalance"]
        row["spread"] = base["spread"]

        # diffs for first N columns
        for c in self.diff_cols:
            prev = self.prev_vals.get(c, base[c])
            row[f"d_{c}"] = base[c] - prev

        # rolling mean/std
        for c in self.roll_cols:
            m, s = self._mean_std(self.roll_bufs[c])
            row[f"m{self.win}_{c}"] = m
            row[f"s{self.win}_{c}"] = s

        for c in ("imbalance", "spread"):
            m, _ = self._mean_std(self.roll_bufs[c])
            row[f"m{self.win}_{c}"] = m

        # update prev_vals after using them
        for c in self.diff_cols:
            self.prev_vals[c] = base[c]

        # Create model input with feature names
        x = np.array([row[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)
        dmat = xgb.DMatrix(x, feature_names=self.feature_names)

        p0 = float(self.model_t0.predict(dmat)[0])
        p1 = float(self.model_t1.predict(dmat)[0])

        pred = np.array([p0, p1], dtype=np.float32)
        pred = np.clip(pred, -6.0, 6.0)
        return pred


if __name__ == "__main__":
    test_file = os.path.join(CURRENT_DIR, "competition_package", "datasets", "valid.parquet")
    model = PredictionModel()
    scorer = ScorerStepByStep(test_file)

    print("Scoring XGB solution on valid.parquet (Weighted Pearson)...")
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
    for t in scorer.targets:
        print(f"  {t}: {results[t]:.6f}")