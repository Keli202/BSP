#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose
-------
FPN variant (with-highlight): read images + nnU-Net predicted masks from
`nnunet/new_dataset_pred/{train,val,test}/{images,masks_pred}`, brighten
mask>0 regions, and pack NPZ per case:
  - images: (T,H,W) float32  [highlighted]
  - labels: (T,) int {0,1}   [merge 1/2 -> 1]
Also saves small PNG previews for a few train cases.

Notes
-----
Earlier in the project, GT labels were accidentally overwritten by predicted
masks and later restored by another script. This file does NOT change that
logic; it only reads current predictions and produces highlighted inputs.

Path Policy (repo-friendly)
---------------------------
Defaults derive from PROJECT_ROOT:
  BASE_NNUNET = $PROJECT_ROOT/nnunet
  DATASET_ROOT = BASE_NNUNET/new_dataset_pred
  OUTPUT_ROOT  = $PROJECT_ROOT/data/processed_features
  VIS_ROOT     = BASE_NNUNET/debug_vis_pred
You can override with env vars: PROJECT_ROOT.
"""

import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ---------- repo-friendly paths ----------
REPO_ROOT   = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
BASE_NNUNET = REPO_ROOT / "nnunet"
DATASET_ROOT = str(BASE_NNUNET / "new_dataset_pred")   # {split}/{images,masks_pred}
OUTPUT_ROOT  = str(REPO_ROOT / "data" / "processed_features")   # .npz out
VIS_ROOT     = str(BASE_NNUNET / "debug_vis_pred")              # preview PNGs

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(VIS_ROOT, exist_ok=True)

# ---------- runtime args (kept) ----------
USE_SMALL_SUBSET = False
SUBSET_RATIO     = 1.0
MAX_WORKERS      = min(8, os.cpu_count() or 1)

# visualization (train split): sample 4 fg cases, 2 frames each
VIS_ENABLE       = True
VIS_SPLIT        = "train"
VIS_NUM_CASES    = 4
VIS_FRAMES_PER   = 2
HIGHLIGHT_FACTOR = 0.15   # brighten toward 255 where mask>0

# ---------- I/O ----------
def load_mha(path):
    itk = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(itk).astype(np.float32)  # (T,H,W)

def list_cases(images_dir, masks_dir):
    names = [f for f in os.listdir(images_dir) if f.endswith(".mha")]
    names = sorted([f for f in names if os.path.exists(os.path.join(masks_dir, f))])
    return names

# ---------- labels: merge 1/2 -> 1 ----------
def get_frame_labels(mask_array):
    """
    A frame is labeled 1 if any pixel equals 1 or 2; otherwise 0.
    """
    T = mask_array.shape[0]
    labels = np.zeros(T, dtype=int)
    for i in range(T):
        frame = mask_array[i]
        if np.any(frame == 1) or np.any(frame == 2):
            labels[i] = 1
        else:
            labels[i] = 0
    return labels

# ---------- highlight ----------
def enhance_images(images, masks, highlight_factor=0.15):
    """
    Grayscale brightening (no color): for mask>0 pixels,
    new = old*(1-factor) + 255*factor
    """
    out = images.copy()
    out = np.clip(out, 0, 255)
    sel = masks > 0
    if sel.any():
        out[sel] = out[sel] * (1.0 - highlight_factor) + 255.0 * highlight_factor
    return out.astype(np.float32)

# ---------- main worker logic (unchanged) ----------
def process_video(image_path, mask_path):
    images = load_mha(image_path)
    masks  = load_mha(mask_path)
    labels = get_frame_labels(masks)          # frame-level 0/1
    enhanced_images = enhance_images(images, masks, highlight_factor=HIGHLIGHT_FACTOR)
    return enhanced_images, labels

def _worker_npz(args):
    filename, img_dir, msk_dir, out_dir = args
    img_path = os.path.join(img_dir, filename)
    msk_path = os.path.join(msk_dir, filename)
    save_npz = os.path.join(out_dir, filename.replace(".mha", ".npz"))
    try:
        images, labels = process_video(img_path, msk_path)
        np.savez_compressed(save_npz, images=images, labels=labels)
        return None
    except Exception as e:
        with open(os.path.join(out_dir, "process_errors.txt"), "a") as ferr:
            ferr.write(f"{filename}\t{repr(e)}\n")
        return filename

# ---------- visualization helpers (unchanged) ----------
def _to_u8_for_show(vol):
    v = vol.astype(np.float32)
    m = np.isfinite(v)
    if not m.any():
        return np.zeros_like(v, dtype=np.uint8)
    lo, hi = np.percentile(v[m], [1, 99])
    if hi <= lo: hi = lo + 1.0
    x = np.clip((v - lo) / (hi - lo), 0, 1)
    return (x * 255.0).round().astype(np.uint8)

def _choose_topk_frames(mask_vol, k=2):
    T = mask_vol.shape[0]
    areas = (mask_vol > 0).reshape(T, -1).sum(1)
    if areas.max() == 0:
        return []
    return list(np.argsort(-areas)[:k])

def _vis_case(save_png, img_vol, pred_vol, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = len(frames)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))
    if cols == 1:
        axes = [axes]

    for ax, z in zip(axes, frames):
        base = _to_u8_for_show(img_vol[z])
        highlight = enhance_images(base, (pred_vol[z] > 0).astype(np.uint8),
                                   highlight_factor=HIGHLIGHT_FACTOR).astype(np.uint8)
        ax.imshow(highlight, cmap="gray")
        ax.set_title(f"z={z}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_png, dpi=120)
    plt.close()

# ---------- entry ----------
def main():
    # 1) pack NPZs per split (no re-splitting)
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        msk_dir = os.path.join(DATASET_ROOT, split, "masks_pred")
        out_dir = os.path.join(OUTPUT_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)

        files = list_cases(img_dir, msk_dir)
        total = len(files)
        if USE_SMALL_SUBSET:
            keep = max(1, int(total * SUBSET_RATIO))
            files = files[:keep]
            print(f"[{split}] small subset: {keep}/{total}")
        else:
            print(f"[{split}] total: {total}")

        args = [(fn, img_dir, msk_dir, out_dir) for fn in files]
        fails = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for res in tqdm(ex.map(_worker_npz, args), total=len(args), desc=f"[{split}] npz"):
                if res is not None:
                    fails.append(res)
        if fails:
            print(f"[WARN] {split}: {len(fails)} files failed (see process_errors.txt)")
        else:
            print(f"[OK] {split}: all npz done.")

    # 2) simple previews from train split
    if VIS_ENABLE:
        split   = VIS_SPLIT
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        msk_dir = os.path.join(DATASET_ROOT, split, "masks_pred")
        vis_dir = os.path.join(VIS_ROOT, split)
        os.makedirs(vis_dir, exist_ok=True)

        files = list_cases(img_dir, msk_dir)

        # pick cases with foreground
        fg_cases = []
        for fn in tqdm(files, desc=f"[{split}] scan FG"):
            pred = load_mha(os.path.join(msk_dir, fn))
            if (pred > 0).any():
                fg_cases.append(fn)
            if len(fg_cases) >= VIS_NUM_CASES:
                break

        for fn in fg_cases:
            cid  = fn[:-4]
            imgs = load_mha(os.path.join(img_dir, fn))
            pred = load_mha(os.path.join(msk_dir, fn))
            frames = _choose_topk_frames(pred, k=VIS_FRAMES_PER)
            if not frames:
                continue
            save_png = os.path.join(vis_dir, f"{cid}.png")
            _vis_case(save_png, imgs, pred, frames)
        print(f"[INFO] PNG saved to: {vis_dir}")

    print("[DONE] NPZs built (images highlighted, labels merged 0/1). Train previews saved.")

if __name__ == "__main__":
    main()
