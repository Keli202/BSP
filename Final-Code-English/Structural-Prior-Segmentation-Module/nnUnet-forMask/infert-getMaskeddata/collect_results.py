#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Purpose:
#   Structural Prior Segmentation Module – step 3
#   Collect nnU-Net predictions, resample to image geometry if needed,
#   copy metadata, and export to a clean repo dataset layout:
#     nnunet/new_dataset_pred/{train,val,test}/{images,masks_pred}
#   For train, also back up original GT to nnunet/new_dataset_pred/train/masks_gt (for reference only).
#
# Important note about GT:
#   Ground-truth labels should remain the official GT. Earlier in the project
#   I mistakenly overwrote GT with inferred masks, which was later corrected
#   by a separate script to restore the original GT. This script does not change
#   that logic; it only writes predicted masks to masks_pred and optionally
#   backs up the existing GT for reference in the train split.
import os
import random
import json
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ---------- Path configuration (repo-friendly) ----------
REPO_ROOT   = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE_NNUNET = REPO_ROOT / "nnunet"
DATA_DIR    = BASE_NNUNET / "data"

SPLIT_DIR   = str(DATA_DIR / "splits_from_npz")
TMP_IMG_DIR = str(DATA_DIR / "tmp_for_infer" / "images")   # *_0000.mha
TMP_MSK_DIR = str(DATA_DIR / "tmp_for_infer" / "masks")    # original GT for reference
INFER_OUT   = str(DATA_DIR / "infer_results")              # raw nnU-Net outputs

FINAL_DIR   = str(BASE_NNUNET / "new_dataset_pred")        # consolidated dataset
os.makedirs(FINAL_DIR, exist_ok=True)

RANDOM_SEED = 42
MAX_CHECK_PER_SPLIT = 5   # sampling cap for quick checks
SAVE_PROBLEM_LIST = True  # save missing/failed cases for review

# ---------- Utilities ----------
def read_ids(split: str) -> List[str]:
    p = os.path.join(SPLIT_DIR, f"{split}_list.txt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing split list: {p}")
    with open(p) as f:
        return [x.strip() for x in f if x.strip()]

def same_header(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize()==b.GetSize() and
            a.GetSpacing()==b.GetSpacing() and
            a.GetOrigin()==b.GetOrigin() and
            a.GetDirection()==b.GetDirection())

def copy_all_metadata(dst: sitk.Image, src: sitk.Image):
    """Copy MetaData dictionary keys/values from src to dst (geometry already handled by CopyInformation)."""
    try:
        keys = list(src.GetMetaDataKeys())
    except Exception:
        keys = []
    for k in keys:
        try:
            dst.SetMetaData(k, src.GetMetaData(k))
        except Exception:
            pass  # ignore incompatible keys

def ref_image_for_metadata(case_id: str, split: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (ref_msk_path, ref_img_path).
    Prefer GT mask as metadata source (some datasets carry label codes in mask MetaData).
    If GT is absent, fall back to the image.
    """
    msk_gt = os.path.join(TMP_MSK_DIR, split, f"{case_id}.mha")
    img    = os.path.join(TMP_IMG_DIR, split, f"{case_id}_0000.mha")
    return (msk_gt if os.path.exists(msk_gt) else None,
            img if os.path.exists(img) else None)

def resample_like(pred: sitk.Image, ref: sitk.Image) -> sitk.Image:
    if same_header(pred, ref):
        return pred
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)   # nearest for label maps
    rs.SetDefaultPixelValue(0)
    return rs.Execute(pred)

def find_prediction_path(split: str, cid: str) -> Optional[str]:
    """
    Find prediction file for cid under INFER_OUT/{split}. Supported endings:
      - exact: cid.nii.gz / cid.mha / cid.nii
      - fallback: first file that starts with cid and has an allowed extension
    """
    out_dir = os.path.join(INFER_OUT, split)
    if not os.path.isdir(out_dir):
        return None

    exts = [".nii.gz", ".mha", ".nii"]
    # exact match first
    for ext in exts:
        p = os.path.join(out_dir, f"{cid}{ext}")
        if os.path.exists(p):
            return p

    # fallback by prefix
    cands = []
    for name in os.listdir(out_dir):
        if name.startswith(cid) and (name.endswith(".nii.gz") or name.endswith(".mha") or name.endswith(".nii")):
            cands.append(name)
    cands.sort()
    if cands:
        return os.path.join(out_dir, cands[0])

    return None

def _labels_preview(arr: np.ndarray, max_items: int = 6) -> str:
    """Compact label set preview for logging."""
    uniq = np.unique(arr)
    if uniq.size > max_items:
        head = ", ".join(map(str, uniq[:max_items]))
        return f"{{{head}, ...}} (total {uniq.size})"
    return "{" + ", ".join(map(str, uniq)) + "}"

def write_mask_with_metadata(pred_path: str, ref_msk_path: Optional[str],
                             ref_img_path: str, out_path: str) -> Tuple[bool, str]:
    """
    Write predicted mask as .mha with geometry/metadata aligned to the reference image:
      1) Resample prediction to the reference image geometry if necessary.
      2) CopyInformation(ref_img) to match size/spacing/origin/direction.
      3) Copy MetaData from GT mask if available, otherwise from the image.
      4) Force uint8 and keep original label values (no remapping).
    """
    try:
        pred = sitk.ReadImage(pred_path)
        ref_img = sitk.ReadImage(ref_img_path)
        pred = resample_like(pred, ref_img)

        preview = _labels_preview(sitk.GetArrayFromImage(pred))

        if pred.GetPixelID() != sitk.sitkUInt8:
            pred = sitk.Cast(pred, sitk.sitkUInt8)

        pred.CopyInformation(ref_img)

        try:
            ref_meta_src = sitk.ReadImage(ref_msk_path) if ref_msk_path else ref_img
        except Exception:
            ref_meta_src = ref_img
        copy_all_metadata(pred, ref_meta_src)

        sitk.WriteImage(pred, out_path)
        return True, f"OK; labels={preview}"
    except Exception as e:
        return False, f"ERROR: {e}"

# ---------- Main pipeline ----------
def merge_split(split: str, log: dict):
    ids = read_ids(split)

    img_out = os.path.join(FINAL_DIR, split, "images")
    msk_out = os.path.join(FINAL_DIR, split, "masks_pred")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    # Back up train GT for reference only (not used to overwrite anything)
    gt_out = None
    if split == "train":
        gt_out = os.path.join(FINAL_DIR, "train", "masks_gt")
        os.makedirs(gt_out, exist_ok=True)

    ok, miss_img, miss_pred, fail_write = 0, 0, 0, 0
    missing_images, missing_preds, write_fails = [], [], []

    for cid in ids:
        # Image reference geometry (from *_0000.mha prepared in step 1)
        img_src = os.path.join(TMP_IMG_DIR, split, f"{cid}_0000.mha")
        if not os.path.exists(img_src):
            miss_img += 1
            missing_images.append(cid)
            print(f"[WARN] {split}/{cid} missing image (tmp *_0000.mha): {img_src}")
            continue

        # Prediction path
        pred = find_prediction_path(split, cid)
        if pred is None:
            miss_pred += 1
            missing_preds.append(cid)
            print(f"[WARN] {split}/{cid} missing prediction (.nii/.nii.gz/.mha)")
            # Still copy the image to keep images directory complete
            try:
                img_itk = sitk.ReadImage(img_src)
                sitk.WriteImage(img_itk, os.path.join(img_out, f"{cid}.mha"))
            except Exception as e:
                print(f"[WARN] {split}/{cid} copy image fallback failed: {e}")
            continue

        # Copy image as {cid}.mha (preserve header + MetaData)
        img_itk = sitk.ReadImage(img_src)
        sitk.WriteImage(img_itk, os.path.join(img_out, f"{cid}.mha"))

        # Metadata source (prefer GT mask if present)
        ref_msk, ref_img = ref_image_for_metadata(cid, split)
        if ref_img is None:
            # Hard fallback: if reference image is missing, use the just-copied output image
            ref_img = os.path.join(img_out, f"{cid}.mha")

        # Write aligned predicted mask
        ok_write, msg = write_mask_with_metadata(pred, ref_msk, ref_img,
                                                 os.path.join(msk_out, f"{cid}.mha"))
        if not ok_write:
            fail_write += 1
            write_fails.append((cid, msg))
            print(f"[ERROR] {split}/{cid} write mask failed: {msg}")
            continue

        # Backup train GT for reference
        if split == "train" and ref_msk and os.path.exists(ref_msk):
            try:
                sitk.WriteImage(sitk.ReadImage(ref_msk), os.path.join(gt_out, f"{cid}.mha"))
            except Exception as e:
                print(f"[WARN] backup GT failed for {cid}: {e}")

        ok += 1

    log[split] = {
        "total": len(ids),
        "ok": ok,
        "missing_image": miss_img,
        "missing_pred": miss_pred,
        "write_fail": fail_write
    }
    print(f"[INFO] {split} summary: total={len(ids)} ok={ok} missing_img={miss_img} missing_pred={miss_pred} write_fail={fail_write}")

    # Problem lists for later review/reruns
    if SAVE_PROBLEM_LIST:
        prob_dir = os.path.join(FINAL_DIR, "_problems")
        os.makedirs(prob_dir, exist_ok=True)
        def write_list(name, items):
            with open(os.path.join(prob_dir, f"{split}_{name}.txt"), "w") as f:
                for x in items:
                    f.write(f"{x}\n")
        write_list("missing_images", missing_images)
        write_list("missing_predictions", missing_preds)
        with open(os.path.join(prob_dir, f"{split}_write_fail.txt"), "w") as f:
            for cid, m in write_fails:
                f.write(f"{cid}\t{m}\n")

def quick_checks():
    """
    Quick integrity checks:
      1) Sample up to MAX_CHECK_PER_SPLIT cases with GT per split and print header match and Dice(>0).
      2) Full header audit: report first 20 mismatches in geometry.
    """
    random.seed(RANDOM_SEED)

    def header_equal(a,b):
        return same_header(a,b)

    # Sampled checks
    for sp in ["train","val","test"]:
        pred_dir = os.path.join(FINAL_DIR, sp, "masks_pred")
        img_dir  = os.path.join(FINAL_DIR, sp, "images")
        gt_dir   = os.path.join(TMP_MSK_DIR, sp)  # original GT, reference only

        if not (os.path.isdir(pred_dir) and os.path.isdir(img_dir)):
            print(f"[Check:{sp}] skip (missing dirs)")
            continue

        gtcases = [f[:-4] for f in os.listdir(gt_dir) if f.endswith(".mha")] if os.path.isdir(gt_dir) else []
        random.shuffle(gtcases)
        sample = gtcases[:MAX_CHECK_PER_SPLIT]

        if not sample:
            print(f"[Check:{sp}] no GT found; skip.")
            continue

        print(f"------ [Check {sp}] sample {len(sample)} cases ------")
        for cid in sample:
            try:
                img = sitk.ReadImage(os.path.join(img_dir,  f"{cid}.mha"))
                msk = sitk.ReadImage(os.path.join(pred_dir, f"{cid}.mha"))
                ok  = header_equal(img, msk)

                gtP = os.path.join(gt_dir, f"{cid}.mha")
                dice = None
                if os.path.exists(gtP):
                    gt = sitk.ReadImage(gtP)
                    # If geometry differs, resample GT to the image geometry for Dice check only
                    if not header_equal(gt, img):
                        rs = sitk.ResampleImageFilter()
                        rs.SetReferenceImage(img)
                        rs.SetInterpolator(sitk.sitkNearestNeighbor)
                        rs.SetDefaultPixelValue(0)
                        gt = rs.Execute(gt)
                    g = sitk.GetArrayFromImage(gt)  > 0
                    p = sitk.GetArrayFromImage(msk) > 0
                    inter = (g & p).sum(); s = g.sum() + p.sum()
                    dice = (2.0*inter/s) if s>0 else 1.0
                print(f"[{sp}] {cid}  header_match={ok}  dice={None if dice is None else round(float(dice),4)}")
                if not ok:
                    print(f"      size(img/msk)={img.GetSize()}/{msk.GetSize()} spacing={img.GetSpacing()}/{msk.GetSpacing()}")
            except Exception as e:
                print(f"[Check:{sp}] {cid} ERROR: {e}")

    # Full geometry audit
    print("\n====== Header audit (full) ======")
    bad = []
    for sp in ["train","val","test"]:
        imgd = os.path.join(FINAL_DIR, sp, "images")
        mskd = os.path.join(FINAL_DIR, sp, "masks_pred")
        if not (os.path.isdir(imgd) and os.path.isdir(mskd)):
            continue
        files = [f for f in os.listdir(imgd) if f.endswith(".mha")]
        for f in files:
            try:
                img = sitk.ReadImage(os.path.join(imgd, f))
                msk = sitk.ReadImage(os.path.join(mskd, f))
                if not same_header(img, msk):
                    bad.append((sp, f,
                                img.GetSize(),   msk.GetSize(),
                                img.GetSpacing(),msk.GetSpacing(),
                                img.GetOrigin(), msk.GetOrigin()))
            except Exception as e:
                bad.append((sp, f, f"ERROR: {e}"))
    print(f"mismatches = {len(bad)}")
    for rec in bad[:20]:
        print("[Bad]", rec)

def main():
    summary = {}
    for sp in ["train", "val", "test"]:
        out_dir = os.path.join(INFER_OUT, sp)
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(f"Missing predictions dir: {out_dir}")
        merge_split(sp, summary)

    # Write summary JSON
    with open(os.path.join(FINAL_DIR, "merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[INFO] merge_summary.json written:", os.path.join(FINAL_DIR, "merge_summary.json"))

    # Quick integrity checks
    quick_checks()

    print("\n[INFO] Done. New dataset at:", FINAL_DIR)
    print("Structure: {train,val,test}/{images,masks_pred} (train also has masks_gt)")
    print("Raw inference outputs:", INFER_OUT)

if __name__ == "__main__":
    main()
