#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ours (VTN/FPN) — Evaluation
---------------------------
Evaluates the FPN model using NPZs from the **GT-corrected** feature set:
  $DATA_ROOT/processed_features_gtlabels/features/{SPLIT}/*.npz

Each NPZ contains:
  - features: (T, L, C), L∈{4,9}, C=768
  - labels  : (T,) in {0,1}

Path overrides:
  PROJECT_ROOT, DATA_ROOT, DATA_DIR, MODEL_PATH, SPLIT
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from BSP_model import MultiLayerFusionTransformerFPN  # FPN model
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from pathlib import Path

# -------- paths --------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

SPLIT = os.getenv("SPLIT", "test")
DEFAULT_DATA_DIR = DATA_ROOT / "processed_features_gtlabels" / "features" / SPLIT

DATA_DIR   = str(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
MODEL_PATH = str(os.getenv("MODEL_PATH", "checkpoints/best_model_fpn.pth"))
SAVE_DIR   = f'eval_output/{SPLIT}'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeadFusionMLP(torch.nn.Module):
    """Keep the exact same fusion MLP as in training."""
    def __init__(self, in_dim=4, hidden_dim=16):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # (B, T)

def load_model(input_dim, levels):
    """Construct model exactly as in training and load weights."""
    model = MultiLayerFusionTransformerFPN(
        feature_dim=input_dim, hidden_dim=256,
        layers_per_stage=1,
        nhead=8, num_ff=512, dropout=0.1,
        use_conv1d=True,
        conv_window=4, causal_conv=True, manual_conv_weights=[0.1, 0.15, 0.25, 0.5],
        use_se=True, se_reduction=16,
        use_bitrans=True
    ).to(device)
    fusion_mlp = HeadFusionMLP(in_dim=4, hidden_dim=16).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state['model'])
    fusion_mlp.load_state_dict(state['fusion_mlp'])
    model.eval(); fusion_mlp.eval()
    return model, fusion_mlp

@torch.no_grad()
def evaluate_sample(npz_path, model, fusion_mlp):
    data = np.load(npz_path)
    feats_np = data['features']           # (T, L, C)
    true_labels = data['labels']          # (T,)
    assert feats_np.ndim == 3, f"Expect (T,L,C), got {feats_np.shape}"

    feats = torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,L,C)
    heads, h4 = model(feats)               # heads:(1,T,4), h4:(1,T,Cf)
    heads = heads.squeeze(0)               # (T,4)
    h4 = h4.squeeze(0)                     # (T,Cf)

    fused_logits = fusion_mlp(heads.unsqueeze(0)).squeeze(0)  # (T,)
    fused_probs  = torch.sigmoid(fused_logits).cpu().numpy()  # (T,)
    head_probs   = torch.sigmoid(heads).cpu().numpy()         # (T,4)
    pred_labels  = (fused_probs > 0.9).astype(int)            # threshold per your code

    return true_labels, pred_labels, fused_probs, head_probs, h4.cpu().numpy()

def compute_cosine_similarity(features, true_labels, pred_labels):
    pred_kf_idx = np.where(pred_labels == 1)[0]
    true_kf_idx = np.where(true_labels == 1)[0]
    if len(true_kf_idx) == 0 or len(pred_kf_idx) == 0:
        return None, None
    pred_kf_feats = features[pred_kf_idx]
    true_kf_feats = features[true_kf_idx]
    S = cosine_similarity(pred_kf_feats, true_kf_feats)
    part1 = S.max(axis=1).sum() / len(pred_kf_feats)
    part2 = S.max(axis=0).sum() / len(true_kf_feats)
    mean_sim = 0.5 * (part1 + part2)
    return mean_sim, S

model = None
fusion_mlp = None
all_true, all_pred = [], []
all_mean_sim = []
all_probs = []
case_count = 0
case_with_keyframe = 0

first_file = next((f for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.npz')), None)
if first_file is None:
    raise FileNotFoundError(f"No .npz files in {DATA_DIR}")
tmp = np.load(os.path.join(DATA_DIR, first_file))
assert tmp['features'].ndim == 3, f"Expect (T,L,C), got {tmp['features'].shape}"
_, L_levels, C_dim = tmp['features'].shape
print(f"[INFO] feature shape per frame: levels={L_levels}, dim={C_dim}")

model, fusion_mlp = load_model(C_dim, L_levels)

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith('.npz'):
        continue
    npz_path = os.path.join(DATA_DIR, file)
    data = np.load(npz_path)
    features_np = data['features']
    assert features_np.shape[1] == L_levels and features_np.shape[2] == C_dim, \
        f"Feature shape mismatch in {file}: got {features_np.shape}, expect (*,{L_levels},{C_dim})"

    true_labels, pred_labels, fused_probs, head_probs, h4 = evaluate_sample(npz_path, model, fusion_mlp)
    mean_sim, sim_matrix = compute_cosine_similarity(h4, true_labels, pred_labels)

    all_true.extend(true_labels.tolist())
    all_pred.extend(pred_labels.tolist())
    all_probs.extend(fused_probs.tolist())
    case_count += 1

    base_name = os.path.splitext(file)[0]
    binc_true = np.bincount(true_labels.astype(int), minlength=2)
    binc_pred = np.bincount(pred_labels.astype(int), minlength=2)
    print(f"{file}: True label count: {binc_true}, Pred label count: {binc_pred}")

    # Per-frame CSV (includes four heads and fused)
    df = pd.DataFrame({
        'frame_idx': np.arange(len(fused_probs)),
        'prob_head1': head_probs[:,0],
        'prob_head2': head_probs[:,1],
        'prob_head3': head_probs[:,2],
        'prob_head4': head_probs[:,3],
        'prob_fused': fused_probs,
        'true_label': true_labels,
        'pred_label': pred_labels
    })
    df.to_csv(os.path.join(SAVE_DIR, f"{base_name}_probs.csv"), index=False)

    # Cosine similarity heatmap (if both sides have keyframes)
    if sim_matrix is not None:
        all_mean_sim.append(mean_sim)
        case_with_keyframe += 1
        plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"Cosine Similarity Heatmap - {base_name}")
        plt.xlabel("True Keyframes"); plt.ylabel("Predicted Keyframes")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_cosine_heatmap.png"))
        plt.clf()
        print(f"[Done] {file} - Mean Cosine Similarity: {mean_sim:.4f}")
    else:
        print(f"[Done] {file} - No keyframes for comparison.")

    # Label comparison (fused)
    plt.figure(figsize=(12, 3))
    plt.plot(true_labels, label='True', color='green', linewidth=1)
    plt.plot(pred_labels, label='Pred(Fused)', color='orange', linestyle='--', linewidth=1)
    plt.title(f"Label Comparison - {base_name}")
    plt.xlabel("Frame Index"); plt.ylabel("Label")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_label_comparison.png"))
    plt.clf()

    # Probability curves
    plt.figure(figsize=(12, 4))
    plt.plot(fused_probs, label='Prob Fused')
    plt.plot(head_probs[:,0], label='Head1', alpha=0.5)
    plt.plot(head_probs[:,1], label='Head2', alpha=0.5)
    plt.plot(head_probs[:,2], label='Head3', alpha=0.5)
    plt.plot(head_probs[:,3], label='Head4', alpha=0.5)
    plt.legend()
    plt.title(f"Fused/Heads Sigmoid Probability - {base_name}")
    plt.xlabel("Frame Index"); plt.ylabel("Probability")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_probs.png"))
    plt.clf()

    np.savez(os.path.join(SAVE_DIR, f"{base_name}_results.npz"),
             true_labels=true_labels,
             pred_labels=pred_labels,
             fused_probs=fused_probs,
             head_probs=head_probs,
             mean_cosine_similarity=mean_sim,
             h4_features=h4)

# -------- Global stats --------
target_names = ['Background', 'Keyframe']
print("\n--- True label distribution (all frames):", Counter(all_true))
print("--- Predicted label distribution (all frames):", Counter(all_pred))

try:
    print("\n--- Classification Report (all frames, threshold=0.9, fused) ---")
    print(classification_report(all_true, all_pred, target_names=target_names, digits=4))
    print("\n--- Confusion Matrix (all frames, threshold=0.9) ---")
    print(confusion_matrix(all_true, all_pred))
except Exception as e:
    print(f"Failed to compute classification report or confusion matrix: {e}")

if all_mean_sim:
    print(f"\n--- Mean cosine similarity (cases with keyframes): {np.mean(all_mean_sim):.4f} ---")
    print(f"Evaluated on {case_with_keyframe} / {case_count} cases.")
else:
    print("\nNo cases with keyframes for cosine similarity calculation.")

# -------- Threshold sweep (fused) --------
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
precisions, recalls, f1s = [], [], []
all_true_arr = np.array(all_true)
all_probs_arr = np.array(all_probs)

for thresh in thresholds:
    preds = (all_probs_arr > thresh).astype(int)
    precisions.append(precision_score(all_true_arr, preds, zero_division=0))
    recalls.append(recall_score(all_true_arr, preds, zero_division=0))
    f1s.append(f1_score(all_true_arr, preds, zero_division=0))

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, marker='o', label="Precision")
plt.plot(thresholds, recalls, marker='s', label="Recall")
plt.plot(thresholds, f1s, marker='^', label="F1-score")
plt.xlabel("Threshold"); plt.ylabel("Score")
plt.title("Precision / Recall / F1-score vs Threshold (Fused)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "prf_vs_thresholds.png"))
plt.close()

print("\n========== Precision/Recall/F1 at Different Thresholds (Fused) ==========")
for t, p, r, f in zip(thresholds, precisions, recalls, f1s):
    print(f"Threshold={t:.2f}: Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

preds_08 = (all_probs_arr > 0.8).astype(int)
print("\n--- Classification Report (all frames, threshold=0.8, fused) ---")
print(classification_report(all_true_arr, preds_08, target_names=target_names, digits=4))
print("\n--- Confusion Matrix (all frames, threshold=0.8, fused) ---")
print(confusion_matrix(all_true_arr, preds_08))
