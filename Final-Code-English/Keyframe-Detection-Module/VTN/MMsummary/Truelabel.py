#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix labels in feature NPZs using original GT masks (do NOT touch features).
---------------------------------------------------------------------------
Background
- In an earlier step, GT labels were accidentally replaced by nnUNet inference
  masks. That affects the per-frame labels stored alongside features.
- This script rebuilds frame labels strictly from **original GT masks**:
      label_t = 1 if mask_t has any foreground (>0), else 0
- It writes a **new** dataset at:
      $DATA_ROOT/processed_features_gtlabels/features/{train,val,test}/
  and does NOT modify the original feature files.

Length Alignment Policy
- If T_feat (frames in features) != T_gt (frames in GT mask), we **only** resample
  the labels (nearest-neighbor on the time axis) to match T_feat.
- Each aligned case is logged to 'aligned_cases.txt' and printed as [OK-ALIGNED].
- If T_feat == T_gt, printed as [OK].

Run:
  python Truelabel.py
"""

import os
import sys
import csv
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path

# -------- Repo-friendly paths (paths only; logic unchanged) --------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

# Input feature root (contains {train,val,test}/*.npz) — original, possibly wrong labels:
DEFAULT_INPUT_FEATURE_ROOT  = DATA_ROOT / "processed_features" / "features"
# Output feature root (new, corrected labels written here):
DEFAULT_OUTPUT_FEATURE_ROOT = DATA_ROOT / "processed_features_gtlabels" / "features"
# Directory with original GT masks (.mha), used to regenerate labels:
DEFAULT_GT_MASK_DIR = DATA_ROOT / "raw_gt_masks"

INPUT_FEATURE_ROOT  = str(os.getenv("INPUT_FEATURE_ROOT",  DEFAULT_INPUT_FEATURE_ROOT))
OUTPUT_FEATURE_ROOT = str(os.getenv("OUTPUT_FEATURE_ROOT", DEFAULT_OUTPUT_FEATURE_ROOT))
GT_MASK_DIR         = str(os.getenv("GT_MASK_DIR",        DEFAULT_GT_MASK_DIR))
SPLITS = ["train", "val", "test"]

# If you want a strict check (optional), set EXPECTED_OK to your expected sample count or None to skip.
EXPECTED_OK = None

# Reports (written beside OUTPUT_FEATURE_ROOT)
OUT_DIR_FOR_REPORTS = os.path.dirname(OUTPUT_FEATURE_ROOT.rstrip("/")) or "."
SAVE_REPORT_PATH   = os.path.join(OUT_DIR_FOR_REPORTS, "fix_report.csv")
SAVE_PROBLEM_LIST  = os.path.join(OUT_DIR_FOR_REPORTS, "problem_files.txt")
SAVE_ALIGNED_LIST  = os.path.join(OUT_DIR_FOR_REPORTS, "aligned_cases.txt")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_gt_labels(mask_path: str) -> np.ndarray:
    """From GT mask (.mha) to per-frame labels: any foreground -> 1 else 0. Shape: (T,)."""
    itk = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(itk).astype(np.float32)  # (T,H,W)
    T = arr.shape[0]
    return (arr.reshape(T, -1) > 0).any(axis=1).astype(np.int32)

def align_labels_to_len(labels_gt: np.ndarray, T_target: int) -> np.ndarray:
    """Nearest-neighbor resample labels from length T_gt to T_target (labels only)."""
    T_gt = len(labels_gt)
    if T_gt == T_target:
        return labels_gt.astype(np.int32)
    idx = np.round(np.linspace(0, T_gt - 1, T_target)).astype(np.int32)
    idx = np.clip(idx, 0, T_gt - 1)
    return labels_gt[idx].astype(np.int32)

def process_one_file(in_npz_path: str, out_npz_path: str, aligned_fh):
    """
    Read input NPZ (features + old labels), regenerate labels from GT mask,
    align length if needed, and write a new NPZ to out_npz_path.
    Returns (status, record) with status in {'ok', 'ok_aligned', 'missing_gt', 'error'}.
    """
    rec = {"file": os.path.basename(in_npz_path), "status": "", "reason": "",
           "T_feat": -1, "T_gt": -1, "pos_old": -1, "pos_new": -1}
    try:
        data = np.load(in_npz_path)
        if "features" not in data.files:
            rec["status"] = "error"; rec["reason"] = "no 'features' in npz"
            print(f"[ERROR] {in_npz_path} -> {rec['reason']}")
            return rec["status"], rec

        feats = data["features"]  # (T, L, C) or (T, 9, 768)
        labels_old = data["labels"] if "labels" in data.files else None
        T_feat = feats.shape[0]
        rec["T_feat"] = int(T_feat)
        if labels_old is not None:
            rec["pos_old"] = int(np.sum(labels_old))

        base = os.path.splitext(os.path.basename(in_npz_path))[0]
        gt_path = os.path.join(GT_MASK_DIR, base + ".mha")
        if not os.path.exists(gt_path):
            rec["status"] = "missing_gt"; rec["reason"] = f"GT not found: {gt_path}"
            print(f"[MISSING] {in_npz_path} -> {rec['reason']}")
            return rec["status"], rec

        labels_gt = load_gt_labels(gt_path)
        T_gt = labels_gt.shape[0]
        rec["T_gt"] = int(T_gt)

        labels_new = align_labels_to_len(labels_gt, T_feat)
        rec["pos_new"] = int(np.sum(labels_new))

        status = "ok" if T_feat == T_gt else "ok_aligned"

        out = {"features": feats, "labels": labels_new.astype(np.int32)}
        for k in data.files:
            if k not in out:
                out[k] = data[k]
        np.savez_compressed(out_npz_path, **out)

        rec["status"] = status
        if status == "ok_aligned":
            print(f"[OK-ALIGNED] {in_npz_path} -> {out_npz_path} (T_feat={T_feat}, T_gt={T_gt})")
            if aligned_fh is not None:
                aligned_fh.write(f"{rec['file']}\tT_feat={T_feat}\tT_gt={T_gt}\n")
        else:
            print(f"[OK] {in_npz_path} -> {out_npz_path} (T={T_feat})")

        if labels_old is not None and rec["pos_old"] != rec["pos_new"]:
            print(f"        pos_frames: {rec['pos_old']} -> {rec['pos_new']}")

        return rec["status"], rec

    except Exception as e:
        rec["status"] = "error"; rec["reason"] = repr(e)
        print(f"[ERROR] {in_npz_path} -> {rec['reason']}")
        return rec["status"], rec

def main():
    if not os.path.isdir(INPUT_FEATURE_ROOT):
        print(f"[FATAL] INPUT_FEATURE_ROOT not found: {INPUT_FEATURE_ROOT}")
        sys.exit(2)
    if not os.path.isdir(GT_MASK_DIR):
        print(f"[FATAL] GT_MASK_DIR not found: {GT_MASK_DIR}")
        sys.exit(2)

    ensure_dir(OUTPUT_FEATURE_ROOT)

    aligned_fh = open(SAVE_ALIGNED_LIST, "w")
    aligned_fh.write("# files that used label length alignment (nearest-neighbor)\n")
    aligned_fh.write("# file\tT_feat\tT_gt\n")

    report_rows = []
    total_files = 0

    for split in SPLITS:
        in_split_dir  = os.path.join(INPUT_FEATURE_ROOT, split)
        out_split_dir = os.path.join(OUTPUT_FEATURE_ROOT, split)
        ensure_dir(out_split_dir)

        if not os.path.isdir(in_split_dir):
            print(f"[WARN] split not found: {in_split_dir}")
            continue

        files = sorted([f for f in os.listdir(in_split_dir) if f.endswith(".npz")])
        total_files += len(files)
        print(f"[INFO] {split}: {len(files)} files | IN={in_split_dir} -> OUT={out_split_dir}")

        for f in tqdm(files, desc=f"[{split}] copy+relabel"):
            in_npz  = os.path.join(in_split_dir, f)
            out_npz = os.path.join(out_split_dir, f)
            status, rec = process_one_file(in_npz, out_npz, aligned_fh)
            rec["split"] = split
            report_rows.append(rec)

    aligned_fh.close()

    stats = {}
    for r in report_rows:
        stats[r["status"]] = stats.get(r["status"], 0) + 1

    ok_count = stats.get("ok", 0) + stats.get("ok_aligned", 0)
    problems = [r for r in report_rows if r["status"] in ("missing_gt", "error")]

    print("\n========== SUMMARY ==========")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"total npz seen: {total_files}")
    print(f"OK total (ok + ok_aligned): {ok_count}")
    print(f"IN : {INPUT_FEATURE_ROOT}")
    print(f"OUT: {OUTPUT_FEATURE_ROOT}")
    print(f"GT : {GT_MASK_DIR}")
    print(f"[INFO] aligned cases list -> {SAVE_ALIGNED_LIST}")

    if SAVE_REPORT_PATH:
        fieldnames = ["split","file","status","reason","T_feat","T_gt","pos_old","pos_new"]
        with open(SAVE_REPORT_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in report_rows:
                w.writerow({k: r.get(k,"") for k in fieldnames})
        print(f"[INFO] report saved to {SAVE_REPORT_PATH}")

    if SAVE_PROBLEM_LIST:
        with open(SAVE_PROBLEM_LIST, "w") as f:
            if not problems:
                f.write("# no problems found\n")
            else:
                f.write("# status\t split\t file\t reason\n")
                for r in problems:
                    f.write(f"{r['status']}\t{r.get('split','')}\t{r['file']}\t{r.get('reason','')}\n")
        print(f"[INFO] problem list saved to {SAVE_PROBLEM_LIST}")

    if EXPECTED_OK is not None and ok_count != EXPECTED_OK:
        print(f"[FAIL] OK count {ok_count} != EXPECTED_OK {EXPECTED_OK}")
        sys.exit(1)

    if problems:
        print(f"[FAIL] Found {len(problems)} problem files (missing_gt/error).")
        sys.exit(1)

    print("[DONE] All good.")
    sys.exit(0)

if __name__ == "__main__":
    main()
