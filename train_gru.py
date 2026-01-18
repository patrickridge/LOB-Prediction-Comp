import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("competition_package/datasets")
TRAIN_FILE = DATA_DIR / "train.parquet"
VALID_SMALL_FILE = DATA_DIR / "valid_small.parquet"

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class SeqDataset(Dataset):
    """
    Returns full sequences (X: [T, 32], y: [T, 2], need_pred: [T])
    """
    def __init__(self, df: pd.DataFrame):
        self.groups = []
        for seq_ix, g in df.groupby("seq_ix", sort=False):
            g = g.sort_values("step_in_seq")
            x = g.iloc[:, 3:35].to_numpy(dtype=np.float32)     # 32 feats
            y = g.iloc[:, 35:37].to_numpy(dtype=np.float32)    # 2 targets
            need = g["need_prediction"].to_numpy(dtype=bool)
            self.groups.append((x, y, need))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        x, y, need = self.groups[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(need.astype(np.uint8))


def collate_pad(batch):
    # All sequences are length 1000 in your data, so we can just stack.
    xs, ys, needs = zip(*batch)
    X = torch.stack(xs, dim=0)       # [B, T, 32]
    Y = torch.stack(ys, dim=0)       # [B, T, 2]
    N = torch.stack(needs, dim=0)    # [B, T]
    return X, Y, N


class GRUModel(nn.Module):
    def __init__(self, d_in=32, hidden=128, d_out=2):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, d_out)

    def forward(self, x):
        # x: [B, T, 32]
        h, _ = self.gru(x)           # [B, T, H]
        out = self.head(h)           # [B, T, 2]
        return out


def weighted_mse(pred, target):
    # weights = |target| (per component) -> average across components
    w = target.abs().clamp_min(1e-3)
    return ((pred - target) ** 2 * w).mean()


def main():
    print("Loading...")
    train_df = pd.read_parquet(TRAIN_FILE)
    valid_df = pd.read_parquet(VALID_SMALL_FILE)

    train_ds = SeqDataset(train_df)
    valid_ds = SeqDataset(valid_df)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_pad)
    valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_pad)

    model = GRUModel(hidden=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train only on scored region (need_prediction==True)
    for epoch in range(3):
        model.train()
        total = 0.0
        for X, Y, N in train_loader:
            X, Y, N = X.to(DEVICE), Y.to(DEVICE), N.to(DEVICE)
            pred = model(X)

            mask = (N == 1)          # [B, T]
            pred_s = pred[mask]      # [N_scored, 2]
            y_s = Y[mask]            # [N_scored, 2]

            loss = weighted_mse(pred_s, y_s)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for X, Y, N in valid_loader:
                X, Y, N = X.to(DEVICE), Y.to(DEVICE), N.to(DEVICE)
                pred = model(X)
                mask = (N == 1)          # [B, T]
                pred_s = pred[mask]      # [N_scored, 2]
                y_s = Y[mask]            # [N_scored, 2]
                vtotal += weighted_mse(pred_s, y_s).item()

        print(f"epoch {epoch+1}: train_loss={total/len(train_loader):.4f} valid_loss={vtotal/len(valid_loader):.4f}")

    out_path = ART_DIR / "gru.pt"
    torch.save({"state_dict": model.state_dict()}, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()