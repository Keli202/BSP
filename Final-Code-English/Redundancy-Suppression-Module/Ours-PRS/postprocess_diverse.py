#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redundancy Suppression (Ours / FPN-VTN version)

- Input results: per-case *_results.npz produced by Ours (FPN/VTN) evaluation.
  You can point to the directory via env OURS_RESULTS_DIR. Default: "eval_output/test".
- GT masks (.mha) are used only for 0/1/2 frame-level evaluation.
  Override via env GT_MASK_DIR.

Processing pipeline (document only; implementation unchanged):
  1) Probability smoothing with "peak-preserving" guard (SG + EMA, with downward clamp),
  2) Thresholding -> segments (remove isolated peaks, merge small gaps),
  3) Peak rescue (short but sharp segments validated by feature cosine),
  4) Segment labeling: interior=1 (key), borders=2 (sub); very short segments (< SMALL_SEG_ALL_KEY_LEN) are fully labeled as 1.

Outputs:
- SAVE_DIR (env OURS_SAVE_DIR or default "eval_output/test_ablation_new")
  ├─ plots_threshold0p9/*.png
  ├─ txt_threshold0p9/*.txt           # per-frame detailed TSV
  ├─ per_case/*.npz                   # post-filter artifacts
  ├─ pred012_txt/*_pred012.txt        # per-frame compact 0/1/2 predictions
  ├─ metrics_five.csv                 # Precision@1 / Recall@1 / F1@1 / WFSS-top1 / WFSS-pred1
  └─ cosine_similarity_summary.csv
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from scipy.signal import savgol_filter, find_peaks

# ================== Paths (env-overridable) ==================
RESULTS_DIR = os.getenv("OURS_RESULTS_DIR", "eval_output/test")                  # where *_results.npz live (Ours)
SAVE_DIR    = os.getenv("OURS_SAVE_DIR",    "eval_output/test_ablation_new")     # output dir for this module
MASK_DIR    = os.getenv(
    "GT_MASK_DIR",
    "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen"
)
os.makedirs(SAVE_DIR, exist_ok=True)

PLOT_DIR     = os.path.join(SAVE_DIR, "plots_threshold0p9")
TXT_DIR      = os.path.join(SAVE_DIR, "txt_threshold0p9")
PERCASE_DIR  = os.path.join(SAVE_DIR, "per_case")
PRED012_DIR  = os.path.join(SAVE_DIR, "pred012_txt")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TXT_DIR,  exist_ok=True)
os.makedirs(PERCASE_DIR, exist_ok=True)
os.makedirs(PRED012_DIR, exist_ok=True)

# ================== Globals / thresholds (document-only) ==================
NEW_BIN_THRESHOLD = 0.9          # main binary threshold for segmenting candidates
ENSURE_AT_LEAST_ONE_1 = False    # keep original behavior (skip pred1 if no positives)
SMALL_SEG_ALL_KEY_LEN = 5        # segments shorter than this are fully labeled as 1

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def get_probs_and_features(data):
    """
    Robustly read fused probabilities and features from *_results.npz.
    Priority:
      - fused probs keys among: ['prob_fused','fused_probs','probs_fused','fused_prob','fused','p_fused']
      - otherwise: mean of head probs/logits (apply sigmoid if needed)
    Features priority: ['h4_features','features','feats','embeddings']
    """
    fused_keys = ['prob_fused','fused_probs','probs_fused','fused_prob','fused','p_fused']
    probs, src = None, None
    for k in fused_keys:
        if k in data:
            arr = np.asarray(data[k]).squeeze()
            if arr.ndim == 1:
                probs = arr.astype(np.float32); src = k; break
    if probs is None:
        head_like = [k for k in data.files if ('head' in k.lower()) and any(s in k.lower() for s in ['prob','logit'])]
        cand_1d   = [k for k in data.files if np.asarray(data[k]).ndim == 1]
        cols = []
        for k in (head_like or cand_1d):
            v = np.asarray(data[k]).squeeze()
            if v.ndim != 1: continue
            vv = v.astype(np.float32)
            if (vv.min() < 0.0) or (vv.max() > 1.0): vv = _sigmoid(vv)
            cols.append(vv)
        if len(cols) >= 2:
            probs = np.mean(np.stack(cols, 0), 0).astype(np.float32); src = f"mean({len(cols)} heads)"
        elif 'probs' in data:
            probs = np.asarray(data['probs']).squeeze().astype(np.float32); src = 'probs'
        else:
            raise KeyError("No fused probs and cannot infer from heads.")
    for k in ['h4_features','features','feats','embeddings']:
        if k in data: feats = np.asarray(data[k]); break
    else:
        raise KeyError("No features found.")
    assert probs.ndim==1 and feats.ndim==2 and feats.shape[0]==probs.shape[0]
    return probs, feats, src

# ----- Peak-preserving smoothing (unchanged) -----
SG_WIN, SG_ORDER = 5, 2
EMA_ALPHA = 0.3
DOWN_CLAMP = 0.05  # do not suppress peaks by more than this

def smooth_probs_guarded(probs):
    p = probs.astype(float)
    if len(p) >= SG_WIN and SG_ORDER < SG_WIN:
        p_sg = savgol_filter(p, SG_WIN, SG_ORDER)
    else:
        p_sg = p.copy()
    p_ema = np.empty_like(p_sg); p_ema[0] = p_sg[0]
    for i in range(1, len(p_sg)):
        p_ema[i] = EMA_ALPHA*p_sg[i] + (1-EMA_ALPHA)*p_ema[i-1]
    p_mix = 0.6*p_sg + 0.4*p_ema
    p_thr = np.maximum(p_mix, probs - DOWN_CLAMP)
    return np.clip(p_thr, 0.0, 1.0), p_mix

# ----- RLE + gap merging (unchanged) -----
MIN_RUN_MAIN = 3
MAX_GAP = 2

def rle_segments(mask):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    if len(idx)==0: return []
    runs = idx.reshape(-1,2)
    return [(int(a), int(b)) for a,b in runs]

def merge_small_gaps(segs, max_gap=2):
    if not segs: return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for a,b in segs[1:]:
        if a - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a,b])
    return [tuple(x) for x in merged]

def segments_from_threshold(p_thr, thr=0.9, min_run=3, max_gap=2):
    mask = p_thr >= thr
    segs = [(a,b) for (a,b) in rle_segments(mask) if (b-a)>=min_run]
    segs = merge_small_gaps(segs, max_gap)
    return segs, mask

# ----- Peak rescue (unchanged) -----
PK_PROM = 0.08
PK_MAX_SPAN = 4
PK_TAU_COS = 0.90

def l2n(F):
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F/np.maximum(n,1e-12)

def rescue_short_peaks(p_raw, p_thr, segs, feats,
                       thr=0.9, prominence=PK_PROM, max_span=PK_MAX_SPAN, tau_pk=PK_TAU_COS):
    T = len(p_thr)
    covered = np.zeros(T, dtype=bool)
    for a,b in segs: covered[a:b] = True
    peaks, _ = find_peaks(p_thr, height=thr, prominence=prominence)
    if len(peaks)==0: return segs
    f = l2n(feats)
    added = []
    for pk in peaks:
        if covered[pk]: continue
        l = pk
        while l-1>=0 and (p_thr[l-1]>=thr or (pk-l<max_span and p_thr[l-1]>=thr-0.03)):
            l -= 1
        r = pk+1
        while r<T and (p_thr[r]>=thr or (r-pk<max_span and p_thr[r]>=thr-0.03)):
            r += 1
        cos_max = float(np.max(f[l:r] @ f[pk]))
        if cos_max < tau_pk:
            continue
        if (r-l)>=2 or p_raw[pk] >= 0.97:
            added.append((l,r))
    if not added: return segs
    all_segs = sorted(segs + added)
    merged = []
    for a,b in all_segs:
        if not merged or a>merged[-1][1]:
            merged.append([a,b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [tuple(x) for x in merged]

# ----- Segment labeling (unchanged logic + small-segment rule) -----
BORDER_W_FIXED  = 3
BORDER_RATIO    = 0.15
USE_RATIO_BORDER = True

def label_segments_interior_border(T, segs):
    """
    Labeling:
      - if length < SMALL_SEG_ALL_KEY_LEN -> all 1
      - else: borders=2, interior=1
    Returns:
      pred (0/1/2 per frame), ones (indices with pred==1)
    """
    pred = np.zeros(T, dtype=int)
    ones = []
    for (a, b) in segs:
        L = b - a
        if L < SMALL_SEG_ALL_KEY_LEN:
            pred[a:b] = 1
            ones.extend(range(a, b))
            continue
        w = max(BORDER_W_FIXED, int(np.floor(L * BORDER_RATIO))) if USE_RATIO_BORDER else BORDER_W_FIXED
        w = min(w, max(1, L // 2))
        left_b  = (a, min(b, a + w))
        right_b = (max(a, b - w), b)
        inner_a, inner_b = left_b[1], right_b[0]
        if inner_b > inner_a:
            pred[inner_a:inner_b] = 1
            ones.extend(range(inner_a, inner_b))
        else:
            pred[a:b] = 2
            mid = a + L // 2
            pred[mid] = 1
            ones.append(mid)
        pred[left_b[0]:left_b[1]] = 2
        pred[right_b[0]:right_b[1]] = 2
    return pred, np.array(sorted(set(ones)), dtype=int)

def wfss_top1(selected_idx, probs, gt_multi):
    if len(selected_idx)==0: return None
    t = selected_idx[np.argmax(probs[selected_idx])]
    g = gt_multi[t]
    return 1.0 if g==1 else (0.6 if g==2 else 0.0)

def wfss_pred1(pred_multi, gt_multi):
    pos = np.where(pred_multi==1)[0]
    if len(pos)==0: return None
    g = gt_multi[pos]
    sc = np.where(g==1, 1.0, np.where(g==2, 0.6, 0.0))
    return float(np.mean(sc))

def get_orig_labels(mask_path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    labels = np.zeros(arr.shape[0], dtype=int)
    for i, frame in enumerate(arr):
        if np.any(frame==1): labels[i]=1
        elif np.any(frame==2): labels[i]=2
        else: labels[i]=0
    return labels

print("\n====== thr=0.9 | smoothing -> threshold segments -> peak rescue -> interior=1/borders=2 ======\n")
print(f"[INFO] RESULTS_DIR = {RESULTS_DIR}")
print(f"[INFO] SAVE_DIR    = {SAVE_DIR}")
print(f"[INFO] GT MASK_DIR = {MASK_DIR}")

y_true_all, y_pred_all = [], []
wfss_top1_vals, wfss_pred1_vals = [], []
key_cos_list, sub_cos_list = [], []
all_pred_feats, all_gt_feats = [], []

for file in tqdm(sorted(os.listdir(RESULTS_DIR))):
    if not file.endswith('_results.npz'): continue
    case_id = file.replace('_results.npz', '')
    data = np.load(os.path.join(RESULTS_DIR, file))

    probs, features, prob_src = get_probs_and_features(data)

    p_thr, p_mix = smooth_probs_guarded(probs)
    segs_main, mask = segments_from_threshold(p_thr, NEW_BIN_THRESHOLD, MIN_RUN_MAIN, MAX_GAP)
    segs = rescue_short_peaks(probs, p_thr, segs_main, features,
                              thr=NEW_BIN_THRESHOLD, prominence=PK_PROM,
                              max_span=PK_MAX_SPAN, tau_pk=PK_TAU_COS)
    pred_multi, selected_idx = label_segments_interior_border(len(probs), segs)

    mask_path = os.path.join(MASK_DIR, case_id + '.mha')
    if not os.path.exists(mask_path):
        print(f"[Warn] mask not found for {case_id}, skip."); continue
    gt_multi = get_orig_labels(mask_path)
    if len(gt_multi) != len(pred_multi):
        print(f"[Warn] length mismatch for {case_id}: gt={len(gt_multi)} pred={len(pred_multi)}; skip."); continue

    pd.DataFrame({
        "frame": np.arange(len(probs), dtype=int),
        "prob_raw": probs.astype(float),
        "prob_smooth": p_mix.astype(float),
        "prob_thr": p_thr.astype(float),
        "pred": pred_multi.astype(int),
        "gt": gt_multi.astype(int)
    }).to_csv(os.path.join(TXT_DIR, f"{case_id}.txt"), sep="\t", index=False)

    np.savez(
        os.path.join(PERCASE_DIR, f"{case_id}_postfilter.npz"),
        pred_multi=pred_multi.astype(np.int16),
        probs_raw=probs.astype(np.float32),
        prob_thr=p_thr.astype(np.float32),
        prob_smooth=p_mix.astype(np.float32),
        segments=np.array(segs, dtype=np.int32)
    )
    np.savetxt(
        os.path.join(PRED012_DIR, f"{case_id}_pred012.txt"),
        np.stack([np.arange(len(pred_multi)), pred_multi], axis=1).astype(int),
        fmt="%d",
        delimiter="\t",
        header="frame\tpred012",
        comments=""
    )

    y_true_all.append(gt_multi); y_pred_all.append(pred_multi)
    wt = wfss_top1(selected_idx, probs, gt_multi);   wp = wfss_pred1(pred_multi, gt_multi)
    if wt is not None: wfss_top1_vals.append(wt)
    if wp is not None: wfss_pred1_vals.append(wp)

    p1 = np.where(pred_multi==1)[0]; g1 = np.where(gt_multi==1)[0]
    if len(p1)>0 and len(g1)>0:
        pf, gf = features[p1], features[g1]
        S = cosine_similarity(pf, gf)
        key_cos_list.append(0.5*((S.max(1).sum()/len(pf)) + (S.max(0).sum()/len(gf))))
        all_pred_feats.append(pf); all_gt_feats.append(gf)
    p2 = np.where(pred_multi==2)[0]; g2 = np.where(gt_multi==2)[0]
    if len(p2)>0 and len(g2)>0:
        pf, gf = features[p2], features[g2]
        S = cosine_similarity(pf, gf)
        sub_cos_list.append(0.5*((S.max(1).sum()/len(pf)) + (S.max(0).sum()/len(gf))))

    plt.figure(figsize=(12,3))
    plt.plot(probs, label=f"Raw ({prob_src})", linewidth=1.0)
    plt.plot(p_mix, label="Smoothed (SG+EMA)", linestyle="--", linewidth=1.0)
    plt.plot(p_thr, label="Thresholded-Protected", linestyle="-.", linewidth=1.0)
    idx1 = np.where(pred_multi==1)[0]; idx2 = np.where(pred_multi==2)[0]
    if len(idx1)>0: plt.scatter(idx1, p_thr[idx1], s=18, marker='o', label='Pred=1')
    if len(idx2)>0: plt.scatter(idx2, p_thr[idx2], s=22, marker='x', label='Pred=2')
    plt.axhline(NEW_BIN_THRESHOLD, color='grey', linestyle=':', linewidth=1)
    for (a,b) in segs: plt.axvspan(a, b, alpha=0.08)
    plt.title(f"{case_id} | thr@0.9 | min_run={MIN_RUN_MAIN} gap={MAX_GAP} | rescue(peaks)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{case_id}_prob_compare.png")); plt.close()

# ================== Aggregate & save core metrics ==================
if y_true_all:
    y_true_all = np.concatenate(y_true_all, 0)
    y_pred_all = np.concatenate(y_pred_all, 0)

    print("\n--- 3-class classification report (frame-level, all test frames) ---")
    print(classification_report(y_true_all, y_pred_all,
                                labels=[0,1,2],
                                target_names=['BG(0)','Key(1)','Sub(2)'],
                                digits=4, zero_division=0))
    print("Confusion Matrix (labels 0,1,2):\n", confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2]))

    gt_bin = (y_true_all == 1).astype(int)
    pd_bin = (y_pred_all == 1).astype(int)
    prec1 = precision_score(gt_bin, pd_bin, zero_division=0)
    rec1  = recall_score(gt_bin, pd_bin, zero_division=0)
    f1_1  = f1_score(gt_bin, pd_bin, zero_division=0)
else:
    prec1 = rec1 = f1_1 = 0.0
    print("\n[Warn] No frames aggregated for PR/F1.")

wfss_top1_mean  = float(np.mean(wfss_top1_vals))  if len(wfss_top1_vals)>0  else 0.0
wfss_pred1_mean = float(np.mean(wfss_pred1_vals)) if len(wfss_pred1_vals)>0 else 0.0

print("\n--- WFSS (per-case avg; skipping cases with no pred-1) ---")
print(f"WFSS-top1  : {wfss_top1_mean:.4f} | counted: {len(wfss_top1_vals)}")
print(f"WFSS-pred1 : {wfss_pred1_mean:.4f} | counted: {len(wfss_pred1_vals)}")

pd.DataFrame([{
    "Precision@1": prec1,
    "Recall@1": rec1,
    "F1@1": f1_1,
    "WFSS_top1": wfss_top1_mean,
    "WFSS_pred1": wfss_pred1_mean,
    "WFSS_top1_count": len(wfss_top1_vals),
    "WFSS_pred1_count": len(wfss_pred1_vals),
    "NEW_BIN_THRESHOLD": NEW_BIN_THRESHOLD,
    "MIN_RUN_MAIN": MIN_RUN_MAIN,
    "MAX_GAP": MAX_GAP,
    "DOWN_CLAMP": DOWN_CLAMP,
    "PK_PROM": PK_PROM,
    "PK_MAX_SPAN": PK_MAX_SPAN,
    "PK_TAU_COS": PK_TAU_COS,
}]).to_csv(os.path.join(SAVE_DIR, "metrics_five.csv"), index=False)

mean_key_cos = float(np.mean(key_cos_list)) if key_cos_list else 0.0
mean_sub_cos = float(np.mean(sub_cos_list)) if key_cos_list else 0.0
if all_pred_feats and all_gt_feats:
    all_pred_feats_cat = np.concatenate(all_pred_feats, 0)
    all_gt_feats_cat   = np.concatenate(all_gt_feats, 0)
    S = cosine_similarity(all_pred_feats_cat, all_gt_feats_cat)
    part1 = S.max(1).sum()/len(all_pred_feats_cat)
    part2 = S.max(0).sum()/len(all_gt_feats_cat)
    all_pos_mean_cos = float(0.5*(part1+part2))
else:
    all_pos_mean_cos = 0.0

pd.DataFrame([{
    "Mean Keyframe Cosine (pred1 vs GT1)": mean_key_cos,
    "Mean Sub-Keyframe Cosine (pred2 vs GT2)": mean_sub_cos,
    "All Positive Mean Cosine": all_pos_mean_cos
}]).to_csv(os.path.join(SAVE_DIR, "cosine_similarity_summary.csv"), index=False)

print("\nSaved:")
print(" - metrics_five.csv :", os.path.join(SAVE_DIR, "metrics_five.csv"))
print(" - TSV per-frame    :", TXT_DIR)
print(" - TXT pred012      :", PRED012_DIR)
print(" - NPZ segments     :", PERCASE_DIR)
print("\nDone.")
