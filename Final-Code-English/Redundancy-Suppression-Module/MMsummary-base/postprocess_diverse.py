#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redundancy Suppression (mmsummary version)

- Input results: per-case *_results.npz from the mmsummary pipeline (basic features without highlight).
  You can point to the directory via env MMSUMMARY_RESULTS_DIR. Default: "eval_output/test".
- GT masks (.mha) are used only to compute frame-level 0/1/2 ground-truth labels for evaluation.
  You can override via env GT_MASK_DIR. Default assumes your original dataset layout.

Key thresholds (do not change logic, just documented):
- PROB_THRESHOLD: probability threshold to form candidate set (>= threshold enters candidate set).
- SIMILARITY_TAU: cosine similarity threshold to mark redundancy (> tau means redundant).
- MAX_FRAMES: upper bound on the number of selected keyframes per case.

Outputs:
- SAVE_DIR (env MMSUMMARY_SAVE_DIR or default "eval_output/test_ablation_new")
  ├─ per_case/*.npz         # post-selection prediction and indices for each case
  ├─ frame_metrics_summary.csv
  ├─ core_metrics_summary.csv  # WFSS-top1 / WFSS-pred1 and hyper-params
  ├─ classification_report.txt
  └─ confusion_matrix.png
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

# ================== Paths (repo-friendly; env-overridable) ==================
RESULTS_DIR = os.getenv("MMSUMMARY_RESULTS_DIR", "eval_output/test")            # where *_results.npz live (mmsummary)
SAVE_DIR    = os.getenv("MMSUMMARY_SAVE_DIR",  "eval_output/test_ablation_new") # output dir for this module
MASK_DIR    = os.getenv(
    "GT_MASK_DIR",
    "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen"
)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "per_case"), exist_ok=True)

# ================== Thresholds (document-only; logic unchanged) ==================
# 1) Probability threshold: to admit candidates from binary probs (>= PROB_THRESHOLD enter candidate set)
PROB_THRESHOLD = 0.9

# 2) Cosine similarity threshold for redundancy covering (> SIMILARITY_TAU means redundant)
SIMILARITY_TAU = 0.96

# Limit the number of selected keyframes per case
MAX_FRAMES = 100

# ================== Selection logic (unchanged) ==================
def key_and_redundant_selection(features, probs, tau, prob_threshold, max_frames):
    """
    Steps:
      1) sort frames by descending probability;
      2) stop when prob < prob_threshold;
      3) a frame not covered becomes "keyframe-1"; frames with cosine similarity > tau to it become "redundant-2";
      4) stop after selecting max_frames keyframes.
    Returns:
      selected_idx (keyframe 1 indices), redundant_idx (redundant 2 indices)
    """
    idx_sorted = np.argsort(-probs)
    selected, used = [], np.zeros(len(probs), dtype=bool)
    redundant = set()
    for idx in idx_sorted:
        if probs[idx] < prob_threshold:
            break
        if used[idx]:
            redundant.add(idx)
            continue
        selected.append(idx)
        used[idx] = True
        sims = cosine_similarity(features[idx:idx + 1], features)[0]
        new_redundant = np.where((sims > tau) & (~used))[0]
        redundant.update(new_redundant)
        used = used | (sims > tau)
        if len(selected) >= max_frames:
            break
    return np.array(selected, dtype=int), np.array(list(redundant), dtype=int)

# ================== Read GT 0/1/2 from .mha (unchanged) ==================
def get_orig_labels(mask_path):
    """
    Return per-frame labels 0/1/2 from GT mask:
      contains pixel value 1 -> label 1 (Key)
      contains pixel value 2 -> label 2 (Sub)
      else                  -> label 0 (BG)
    """
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    labels = np.zeros(mask_array.shape[0], dtype=int)
    for i, frame in enumerate(mask_array):
        if np.any(frame == 1):
            labels[i] = 1
        elif np.any(frame == 2):
            labels[i] = 2
        else:
            labels[i] = 0
    return labels

# ================== WFSS on post-selected results (unchanged) ==================
def wfss_top1(selected_idx, probs, gt_multi):
    if len(selected_idx) == 0:
        return 0.0
    top_local = selected_idx[np.argmax(probs[selected_idx])]
    g = gt_multi[top_local]
    return 1.0 if g == 1 else (0.6 if g == 2 else 0.0)

def wfss_pred1(pred_multi, gt_multi):
    pos_idx = np.where(pred_multi == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    gs = gt_multi[pos_idx]
    scores = np.where(gs == 1, 1.0, np.where(gs == 2, 0.6, 0.0))
    return float(np.mean(scores))

# ================== Main (unchanged core flow, clearer prints) ==================
print("\n====== Core evaluation (3-class metrics + WFSS-top1 + WFSS-pred1) ======\n")
print(f"[INFO] RESULTS_DIR = {RESULTS_DIR}")
print(f"[INFO] SAVE_DIR    = {SAVE_DIR}")
print(f"[INFO] GT MASK_DIR = {MASK_DIR}")

y_true_all, y_pred_all = [], []
wfss_top1_list, wfss_pred1_list = [], []
wfss_rows = []
key_cos_list, sub_cos_list = [], []
all_pred_feats, all_gt_feats = [], []

for file in tqdm(sorted(os.listdir(RESULTS_DIR))):
    if not file.endswith('_results.npz'):
        continue
    case_id = file.replace('_results.npz', '')
    path = os.path.join(RESULTS_DIR, file)
    data = np.load(path)

    # Be compatible with both mmsummary and ours (key names)
    probs    = data['fused_probs'] if 'fused_probs' in data else data['probs']
    features = data['h4_features'] if 'h4_features' in data else data['features']

    selected_idx, redundant_idx = key_and_redundant_selection(
        features, probs, tau=SIMILARITY_TAU, prob_threshold=PROB_THRESHOLD, max_frames=MAX_FRAMES
    )

    pred_multi = np.zeros_like(probs, dtype=int)
    pred_multi[redundant_idx] = 2
    pred_multi[selected_idx]  = 1

    mask_path = os.path.join(MASK_DIR, case_id + '.mha')
    if not os.path.exists(mask_path):
        print(f"[Warn] mask not found for {case_id}, skip.")
        continue
    gt_multi = get_orig_labels(mask_path)
    if len(gt_multi) != len(pred_multi):
        print(f"[Warn] length mismatch for {case_id}: gt={len(gt_multi)} pred={len(pred_multi)}; skip.")
        continue

    np.savez(
        os.path.join(SAVE_DIR, "per_case", f"{case_id}_postselect.npz"),
        pred_multi=pred_multi,
        selected_idx=selected_idx,
        redundant_idx=redundant_idx,
        probs=probs
    )

    y_true_all.append(gt_multi)
    y_pred_all.append(pred_multi)

    w_top1  = wfss_top1(selected_idx, probs, gt_multi)
    w_pred1 = wfss_pred1(pred_multi, gt_multi)
    wfss_top1_list.append(w_top1)
    eligible = 1 if np.any(pred_multi == 1) else 0
    if eligible:
        wfss_pred1_list.append(w_pred1)
    wfss_rows.append({
        "case": case_id,
        "WFSS_top1": w_top1,
        "WFSS_pred1": w_pred1,
        "eligible_for_pred1": eligible
    })

    p1 = np.where(pred_multi == 1)[0]; g1 = np.where(gt_multi == 1)[0]
    if len(p1) > 0 and len(g1) > 0:
        pf, gf = features[p1], features[g1]
        S = cosine_similarity(pf, gf)
        part1 = S.max(axis=1).sum() / len(pf)
        part2 = S.max(axis=0).sum() / len(gf)
        key_cos_list.append(0.5 * (part1 + part2))
        all_pred_feats.append(pf); all_gt_feats.append(gf)

    p2 = np.where(pred_multi == 2)[0]; g2 = np.where(gt_multi == 2)[0]
    if len(p2) > 0 and len(g2) > 0:
        pf, gf = features[p2], features[g2]
        S = cosine_similarity(pf, gf)
        part1 = S.max(axis=1).sum() / len(pf)
        part2 = S.max(axis=0).sum() / len(gf)
        sub_cos_list.append(0.5 * (part1 + part2))

# ===== Aggregate & save (unchanged) =====
if y_true_all:
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    acc = accuracy_score(y_true_all, y_pred_all)
    P, R, F1, supp = precision_recall_fscore_support(y_true_all, y_pred_all, labels=[0,1,2], zero_division=0)

    frame_df = pd.DataFrame([{
        "Acc": acc,
        "P_key": P[1], "R_key": R[1], "F1_key": F1[1], "Supp_key": supp[1],
        "P_sub": P[2], "R_sub": R[2], "F1_sub": F1[2], "Supp_sub": supp[2],
    }])
    frame_df.to_csv(os.path.join(SAVE_DIR, "frame_metrics_summary.csv"), index=False)

    with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(
            y_true_all, y_pred_all,
            labels=[0,1,2],
            target_names=['BG(0)','Key(1)','Sub(2)'],
            digits=4, zero_division=0
        ))

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2])
    plt.figure(figsize=(4.2,3.6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (0/1/2)")
    plt.colorbar()
    ticks = np.arange(3)
    plt.xticks(ticks, ["BG","Key","Sub"])
    plt.yticks(ticks, ["BG","Key","Sub"])
    thr = cm.max() * 0.6 if cm.max()>0 else 1
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i,j] > thr else "black")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()
else:
    pd.DataFrame([{
        "Acc": 0.0, "P_key":0.0, "R_key":0.0, "F1_key":0.0, "Supp_key":0,
        "P_sub":0.0, "R_sub":0.0, "F1_sub":0.0, "Supp_sub":0,
    }]).to_csv(os.path.join(SAVE_DIR, "frame_metrics_summary.csv"), index=False)
    print("\n[Warn] No frames aggregated for PR/F1.")

wfss_top1_mean  = float(np.mean(wfss_top1_list))  if wfss_top1_list  else 0.0
eligible_count  = int(np.sum([r["eligible_for_pred1"] for r in wfss_rows])) if wfss_rows else 0
wfss_pred1_mean = float(np.mean([r["WFSS_pred1"] for r in wfss_rows if r["eligible_for_pred1"]==1])) if eligible_count>0 else 0.0

print("\n--- WFSS (per-case average) ---")
print(f"WFSS-top1  (official): {wfss_top1_mean:.4f}")
print(f"WFSS-pred1 (ours)   : {wfss_pred1_mean:.4f}  (eligible cases = {eligible_count})")

core_df = pd.DataFrame([{
    "WFSS_top1": wfss_top1_mean,
    "WFSS_pred1": wfss_pred1_mean,
    "WFSS_pred1_case_count": eligible_count,
    "PROB_THRESHOLD": PROB_THRESHOLD,
    "SIMILARITY_TAU": SIMILARITY_TAU,
    "MAX_FRAMES": MAX_FRAMES,
}])
core_df.to_csv(os.path.join(SAVE_DIR, "core_metrics_summary.csv"), index=False)
pd.DataFrame(wfss_rows).to_csv(os.path.join(SAVE_DIR, "wfss_cases.csv"), index=False)
print("\nCore metrics saved to:", os.path.join(SAVE_DIR, "core_metrics_summary.csv"))

mean_key_cos = float(np.mean(key_cos_list)) if key_cos_list else 0.0
mean_sub_cos = float(np.mean(sub_cos_list)) if key_cos_list else 0.0
if all_pred_feats and all_gt_feats:
    all_pred_feats_cat = np.concatenate(all_pred_feats, axis=0)
    all_gt_feats_cat   = np.concatenate(all_gt_feats, axis=0)
    S = cosine_similarity(all_pred_feats_cat, all_gt_feats_cat)
    part1 = S.max(axis=1).sum() / len(all_pred_feats_cat)
    part2 = S.max(axis=0).sum() / len(all_gt_feats_cat)
    all_pos_mean_cos = float(0.5 * (part1 + part2))
else:
    all_pos_mean_cos = 0.0

pd.DataFrame([{
    "Mean Keyframe Cosine (pred1 vs GT1)": mean_key_cos,
    "Mean Sub-Keyframe Cosine (pred2 vs GT2)": mean_sub_cos,
    "All Positive Mean Cosine": all_pos_mean_cos
}]).to_csv(os.path.join(SAVE_DIR, "cosine_similarity_summary.csv"), index=False)

print("\nDone.")
