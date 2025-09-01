#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMsummary — Training
--------------------
This trainer uses features extracted from the **basic pipeline (no highlight)**.
Make sure you have already run:
  Keyframe-Detection-Module / FeatureExtraction / Feature-extraction-basic
so that the following NPZ files exist:
  $DATA_ROOT/processed_features/features/{train,val}/*.npz
Each NPZ contains:
  - features: (T, D) or (T, 9, 768) depending on your extractor
  - labels:   (T,) in {0,1}

Path Policy
-----------
Defaults assume a repo layout with data under $PROJECT_ROOT/data.
Override with env vars as needed:
  PROJECT_ROOT  (default: repo root inferred from this file)
  DATA_ROOT     (default: $PROJECT_ROOT/data)
  TRAIN_DIR     (optional) absolute path to train NPZ dir
  VAL_DIR       (optional) absolute path to val NPZ dir

Notes
-----
- Logic (model, loss, early stopping, I/O) is unchanged.
- MMsummary is strictly tied to **basic features** (no nnU-Net highlight).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from BSP_model import VideoTransformer
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Reproducibility --------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------- Repo-friendly paths (paths only; logic unchanged) --------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

# Default to BASIC (no-highlight) features
DEFAULT_TRAIN_DIR = DATA_ROOT / "processed_features" / "features" / "train"
DEFAULT_VAL_DIR   = DATA_ROOT / "processed_features" / "features" / "val"

TRAIN_DIR = str(os.getenv("TRAIN_DIR", DEFAULT_TRAIN_DIR))
VAL_DIR   = str(os.getenv("VAL_DIR",   DEFAULT_VAL_DIR))
SAVE_DIR  = "checkpoints"  # stays relative to the script's CWD

EPOCHS = 150
BATCH_SIZE = 4            # kept as-is (not used below; DataLoader uses batch_size=1)
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 20  # early stopping patience

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoFeatureDataset(Dataset):
    """
    Loads per-video NPZ:
      features: (T, D) or (T, 9, 768)
      labels:   (T,)
    The model below consumes (T, D). If your features are (T, 9, 768),
    make sure upstream collapsed them as needed (we keep the original logic here).
    """
    def __init__(self, folder_path):
        self.features = []
        self.labels = []
        for file in sorted(os.listdir(folder_path)):   # keep deterministic order
            if not file.endswith('.npz'):
                continue
            data = np.load(os.path.join(folder_path, file))
            self.features.append(torch.tensor(data['features'], dtype=torch.float32))
            self.labels.append(torch.tensor(data['labels'],   dtype=torch.float32))  # float for BCE
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = VideoFeatureDataset(TRAIN_DIR)
val_dataset   = VideoFeatureDataset(VAL_DIR)

# NOTE: keep batch_size=1 to preserve original training logic
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

input_dim = train_dataset[0][0].shape[-1]
model = VideoTransformer(feature_dim=input_dim).to(device)

# -------- Compute pos_weight from training labels (unchanged) --------
all_labels = []
for _, labels in train_dataset:
    all_labels.extend(labels.cpu().numpy().tolist())
num_pos = (np.array(all_labels) == 1).sum()
num_neg = (np.array(all_labels) == 0).sum()
pos_weight_value = (num_neg / num_pos) if num_pos > 0 else 1.0
print(f"Train set: pos={num_pos}, neg={num_neg}, pos_weight={pos_weight_value:.2f}")

pos_weight = torch.tensor([pos_weight_value]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_path = os.path.join(SAVE_DIR, 'best_model.pth')
no_improve_epochs = 0  # early stopping counter

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    for feats, labels in train_loader:
        feats = feats.squeeze(0).to(device)    # (T, D)
        labels = labels.squeeze(0).to(device)  # (T,)
        logits, _ = model(feats.unsqueeze(0))  # (1, T, 1)
        logits = logits.squeeze(0).squeeze(-1) # (T,)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.squeeze(0).to(device)
            labels = labels.squeeze(0).to(device)
            logits, _ = model(feats.unsqueeze(0))
            logits = logits.squeeze(0).squeeze(-1)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ---- save best model (unchanged) ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        try:
            torch.save(model.state_dict(), best_model_path)
            print(f" Saved best model at {best_model_path}")
        except Exception as e:
            print(f"[Warning] Failed to save model: {e}")
    else:
        no_improve_epochs += 1
        print(f"No improvement in val loss for {no_improve_epochs} epoch(s).")

    # ---- early stopping (unchanged) ----
    if no_improve_epochs >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}. Training terminated.")
        break

# ---- plots & logs (unchanged) ----
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'))

with open(os.path.join(SAVE_DIR, 'train_log.txt'), 'w') as f:
    for i, (t, v) in enumerate(zip(train_losses, val_losses)):
        f.write(f"Epoch {i+1}: Train Loss = {t:.4f}, Val Loss = {v:.4f}\n")
