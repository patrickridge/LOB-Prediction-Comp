import os
import sys
import numpy as np

import torch
import torch.nn as nn
import onnxruntime as ort

# utils live in competition_package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "competition_package"))
from utils import DataPoint, ScorerStepByStep

# NOTE: submission env is CPU-only; local can use mps
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# -------------------------
# GRU (PyTorch) model
# -------------------------
class GRUModel(nn.Module):
    def __init__(self, d_in=32, hidden=256, d_out=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers >= 2 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, d_out)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)      # out: [B, T, H]
        y = self.head(out)           # y:  [B, T, 2]
        return y, h


class GRUPredictor:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        self.hidden = int(ckpt.get("hidden", 256))
        self.num_layers = int(ckpt.get("num_layers", 2))
        self.dropout = float(ckpt.get("dropout", 0.1))

        self.model = GRUModel(
            hidden=self.hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(DEVICE)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.current_seq_ix = None
        self.h = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.h = None

    @torch.no_grad()
    def step(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_seq(data_point.seq_ix)

        x_np = data_point.state.astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)  # [1,1,32]

        y, self.h = self.model(x, self.h)                     # y: [1,1,2]
        pred = y[0, 0].detach().cpu().numpy().astype(np.float32)
        return pred


# -------------------------
# Baseline (ONNX) model
# -------------------------
class ONNXBaselinePredictor:
    def __init__(self, onnx_path: str, window: int = 100):
        self.window = window
        self.current_seq_ix = None
        self.history = []

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.history = []

    def step(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_seq(data_point.seq_ix)

        self.history.append(data_point.state.astype(np.float32, copy=False))

        # Build last-100 window for inference
        win = self.history[-self.window:]
        if len(win) < self.window:
            pad = [np.zeros_like(win[0])] * (self.window - len(win))
            win = pad + win

        data_arr = np.asarray(win, dtype=np.float32)          # [100, 32]
        data_tensor = np.expand_dims(data_arr, axis=0)        # [1, 100, 32]

        ort_inputs = {self.input_name: data_tensor}
        output = self.ort_session.run([self.output_name], ort_inputs)[0]

        # Output shape could be (1,2) or (1,seq,2)
        if output.ndim == 3:
            pred = output[0, -1, :]
        else:
            pred = output[0]

        return pred.astype(np.float32)


# -------------------------
# Combined PredictionModel (submission entrypoint)
# -------------------------
class PredictionModel:
    """
    Ensemble:
      pred = alpha * GRU + (1 - alpha) * baseline_onnx
    """

    def __init__(self, alpha: float = 0.7):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # paths inside submission zip
        ckpt_path = os.path.join(base_dir, "artifacts", "gru_h256_L2_do0.1.pt")
        onnx_path = os.path.join(base_dir, "baseline.onnx")

        self.gru = GRUPredictor(ckpt_path)
        self.base = ONNXBaselinePredictor(onnx_path, window=100)

        self.alpha = float(alpha)

    def predict(self, data_point: DataPoint):
        # Always advance both models so their state stays correct
        p_gru = self.gru.step(data_point)
        p_base = self.base.step(data_point)

        if not data_point.need_prediction:
            return None

        pred = self.alpha * p_gru + (1.0 - self.alpha) * p_base
        pred = np.clip(pred, -6.0, 6.0).astype(np.float32)
        return pred


if __name__ == "__main__":
    test_file = os.path.join(CURRENT_DIR, "competition_package", "datasets", "valid.parquet")
    model = PredictionModel(alpha=0.7)
    scorer = ScorerStepByStep(test_file)

    print("Scoring ensemble on valid.parquet (Weighted Pearson)...")
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
    for t in scorer.targets:
        print(f"  {t}: {results[t]:.6f}")