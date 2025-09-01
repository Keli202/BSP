#!/usr/bin/env python3
# Purpose:
#   Structural Prior Segmentation Module – step 1
#   Generate train/val/test ID lists from existing feature folders and
#   prepare raw ultrasound images for nnU-Net inference by renaming to *_0000.mha.
#
# Notes about the project context:
#   - We train nnU-Net on the training set and then run inference over all splits
#     to obtain segmentation masks used as a "structural prior".
#   - Ground-truth labels (GT) should remain the official GT for learning/evaluation.
#     In an earlier draft I accidentally overwrote GT with inferred masks; this was
#     later fixed by an external script to restore the correct GT. Logic here remains
#     unchanged; this comment just clarifies intended usage and the prior mistake.
#
# Path policy for GitHub:
#   - By default, paths are relative to the repo root and can be overridden via:
#       PROJECT_ROOT: repo root path
#       FEATURE_VARIANT: "plain" (default) or "fpn"
#   - No computation logic has been altered; only comments and path definitions.
import os
import shutil
from pathlib import Path

# ---------- Path configuration (repo-friendly) ----------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
FEATURE_VARIANT = os.getenv("FEATURE_VARIANT", "plain").lower()  # "plain" or "fpn"

# Raw ACOUSLIC data (unchanged content; just path normalization)
images_dir = str(REPO_ROOT / "data" / "acouslic" / "images" / "stacked_fetal_ultrasound")
masks_dir  = str(REPO_ROOT / "data" / "acouslic" / "masks"  / "stacked_fetal_abdomen")

# Split files will be created here from existing processed features
split_dir = str(REPO_ROOT / "nnunet" / "data" / "splits_from_npz")  # train_list.txt / val_list.txt / test_list.txt

# Inference inputs for nnU-Net v2 (keep .mha; image renamed to *_0000.mha; masks copied only for reference)
tmp_img_dir  = str(REPO_ROOT / "nnunet" / "data" / "tmp_for_infer" / "images")
tmp_mask_dir = str(REPO_ROOT / "nnunet" / "data" / "tmp_for_infer" / "masks")

# We derive split IDs from precomputed features to keep consistency with the detection pipeline.
# Choose the feature variant via FEATURE_VARIANT: "plain" or "fpn".
processed_base = str(REPO_ROOT / "data" / "processed_features" / FEATURE_VARIANT)

USE_SYMLINK = True      # Prefer symlinks for speed/space; falls back to copy if not supported
SKIP_IF_EXISTS = True   # Skip if destination exists (helps re-runs)

os.makedirs(tmp_img_dir,  exist_ok=True)
os.makedirs(tmp_mask_dir, exist_ok=True)

def auto_generate_split_from_existing():
    """
    Generate {train,val,test}_list.txt from existing processed feature folders
    to ensure split consistency with downstream tasks.
    """
    mapping = {
        "train": os.path.join(processed_base, "train"),
        "val":   os.path.join(processed_base, "val"),
        "test":  os.path.join(processed_base, "test")
    }

    os.makedirs(split_dir, exist_ok=True)
    for split, path in mapping.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed features split not found: {path}")
        case_ids = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith(".npz")]
        case_ids.sort()
        with open(os.path.join(split_dir, f"{split}_list.txt"), "w") as f:
            for cid in case_ids:
                f.write(f"{cid}\n")
        print(f"[INFO] Generated split file for {split}: {len(case_ids)} cases")

def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if SKIP_IF_EXISTS and os.path.exists(dst):
        return "skip"
    # Remove existing (including broken symlinks)
    if os.path.lexists(dst):
        os.remove(dst)
    if USE_SYMLINK:
        try:
            os.symlink(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"

def prepare_case(case_id, split):
    # Source files
    src_img  = os.path.join(images_dir, f"{case_id}.mha")
    src_mask = os.path.join(masks_dir,  f"{case_id}.mha")

    # Destinations (nnU-Net expects image as *_0000.mha; mask is kept as reference only)
    dst_img  = os.path.join(tmp_img_dir,  split, f"{case_id}_0000.mha")
    dst_mask = os.path.join(tmp_mask_dir, split, f"{case_id}.mha")

    if not os.path.exists(src_img):
        return False, False, "missing_image"

    img_mode = link_or_copy(src_img, dst_img)
    mask_mode = None
    if os.path.exists(src_mask):
        mask_mode = link_or_copy(src_mask, dst_mask)  # reference only; not used by inference
        return True, True, f"img:{img_mode},mask:{mask_mode}"
    else:
        return True, False, f"img:{img_mode},mask:None"

def load_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]

def main():
    # 1) Build split lists from processed features
    auto_generate_split_from_existing()

    # 2) Prepare *_0000.mha for nnU-Net inference; copy masks for reference if present
    for split in ["train", "val", "test"]:
        list_path = os.path.join(split_dir, f"{split}_list.txt")
        cases = load_list(list_path)
        total = len(cases)
        ok, with_mask, miss = 0, 0, 0

        for cid in cases:
            has_img, has_mask, msg = prepare_case(cid, split)
            if not has_img:
                miss += 1
            else:
                ok += 1
                if has_mask:
                    with_mask += 1

        print(f"[INFO] {split}: total={total} ready={ok} with_GT_ref={with_mask} missing_img={miss}")

    print("\n[INFO] Inference inputs are ready (images named as *_0000.mha).")
    print(f"Images by split: {tmp_img_dir}")
    print(f"Reference GT masks by split (optional): {tmp_mask_dir}")
    print("Next: run nnUNetv2_predict using the above images directory as -i.")

if __name__ == "__main__":
    main()
