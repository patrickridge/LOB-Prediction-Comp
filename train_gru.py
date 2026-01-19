import os
import time
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

# Training config (match your earlier reasonable baseline style)
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-3

# Model config (the upgrade)
HIDDEN = 256
NUM_LAYERS = 2
DROPOUT = 0.1  # only applies when NUM_LAYERS >= 2


class SeqDataset(Dataset):
    """
    Returns full sequences (X: [T, 32], y: [T, 2], need_pred: [T])
    """
    def __init__(self, df: pd.DataFrame):
        self.groups = []
        for _, g in df.groupby("seq_ix", sort=False):
            g = g.sort_values("step_in_seq")
            x = g.iloc[:, 3:35].to_numpy(dtype=np.float32)     # 32 feats
            y = g.iloc[:, 35:37].to_numpy(dtype=np.float32)    # 2 targets
            need = g["need_prediction"].to_numpy(dtype=bool)
            self.groups.append((x, y, need))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        x, y, need = self.groups[idx]
        return (
            torch.from_numpy(x),                         # [T, 32]
            torch.from_numpy(y),                         # [T, 2]
            torch.from_numpy(need.astype(np.uint8)),     # [T]
        )


def collate_stack(batch):
    # sequences are fixed length 1000, so stack is fine
    xs, ys, needs = zip(*batch)
    X = torch.stack(xs, dim=0)       # [B, T, 32]
    Y = torch.stack(ys, dim=0)       # [B, T, 2]
    N = torch.stack(needs, dim=0)    # [B, T]
    return X, Y, N


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

    def forward(self, x):
        # x: [B, T, 32]
        h, _ = self.gru(x)          # [B, T, H]
        out = self.head(h)          # [B, T, 2]
        return out


def weighted_mse(pred, target):
    # pred/target: [N, 2]
    w = target.abs().clamp_min(1e-3)
    return ((pred - target) ** 2 * w).mean()


def main():
    print("Loading train + valid_small...")
    train_df = pd.read_parquet(TRAIN_FILE)
    valid_df = pd.read_parquet(VALID_SMALL_FILE)

    train_ds = SeqDataset(train_df)
    valid_ds = SeqDataset(valid_df)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_stack
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_stack
    )

    model = GRUModel(hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"DEVICE={DEVICE} | BATCH_SIZE={BATCH_SIZE} | EPOCHS={EPOCHS} | HIDDEN={HIDDEN} | LAYERS={NUM_LAYERS} | DROPOUT={DROPOUT}")

    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0

        for X, Y, N in train_loader:
            X, Y, N = X.to(DEVICE), Y.to(DEVICE), N.to(DEVICE)
            pred = model(X)  # [B, T, 2]

            mask = (N == 1)          # [B, T]
            pred_s = pred[mask]      # [N_scored, 2]
            y_s = Y[mask]            # [N_scored, 2]

            loss = weighted_mse(pred_s, y_s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for X, Y, N in valid_loader:
                X, Y, N = X.to(DEVICE), Y.to(DEVICE), N.to(DEVICE)
                pred = model(X)

                mask = (N == 1)
                pred_s = pred[mask]
                y_s = Y[mask]

                vloss += weighted_mse(pred_s, y_s).item()

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (EPOCHS - (epoch + 1))

        print(
            f"epoch {epoch+1}/{EPOCHS}: "
            f"train_loss={total_loss/len(train_loader):.4f} "
            f"valid_loss={vloss/len(valid_loader):.4f} "
            f"| time={epoch_time:.1f}s ETA={eta/60:.1f}m"
        )

    out_path = ART_DIR / f"gru_h{HIDDEN}_L{NUM_LAYERS}_do{DROPOUT}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": HIDDEN,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        out_path,
    )
    print("Saved:", out_path)
    print(f"Total train time: {(time.time()-total_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()