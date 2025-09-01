#!/bin/bash
#SBATCH --job-name=nnunet_infer_all_2d
#SBATCH --partition=gpu
#SBATCH --account=COMS031144
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=nnunet_infer_all_2d_%j.out
#SBATCH --error=nnunet_infer_all_2d_%j.err

# Purpose:
#   Structural Prior Segmentation Module – step 2
#   Run nnUNetv2_predict over all splits prepared in step 1 and save raw predictions.
#
# Path policy for GitHub:
#   - Paths derive from PROJECT_ROOT unless overridden before submission:
#       export PROJECT_ROOT=/path/to/repo-root
#       export NNUNET_RESULTS=/path/to/nnUNet_results  # optional
#   - No algorithmic changes; only path normalization and comments.

echo "========== SLURM job started =========="
echo "SLURM script: $(realpath $0)"
echo "Submit dir: $(pwd)"
echo "Date: $(date)"
echo "========================================"

# Environment
. /user/home/ad21083/initConda.sh
conda activate Detect-BSPnnUnet

set -eo pipefail
shopt -s nullglob

echo "Using GPU(s):"
nvidia-smi --query-gpu=name --format=csv

# nnU-Net configuration (consistent with training; 2D & fold 0)
DATASET_ID=300
CFG="2d"
PLANS="nnUNetResEncUNetLPlans"
TRAINER="nnUNetTrainer"
FOLD=0

# Paths (repo-friendly; overridable)
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_NNUNET="${BASE_NNUNET:-$PROJECT_ROOT/nnunet}"
DATA_DIR="$BASE_NNUNET/data"

SPLIT_DIR="$DATA_DIR/splits_from_npz"                 # train_list.txt / val_list.txt / test_list.txt
TMP_IMG_DIR="$DATA_DIR/tmp_for_infer/images"          # prepared *_0000.mha
INFER_OUT="$DATA_DIR/infer_results"                   # raw predictions (.nii.gz)
mkdir -p "$INFER_OUT"/{train,val,test}

# Checkpoint (allow override via CKPT; otherwise use default layout)
export nnUNet_results="${NNUNET_RESULTS:-$BASE_NNUNET/nnUNet_results}"
CKPT="${CKPT:-$nnUNet_results/Dataset300_FetalAbdomen/${TRAINER}__${PLANS}__${CFG}/fold_${FOLD}/checkpoint_best.pth}"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi
echo "[INFO] Using checkpoint: $CKPT"

# Ensure split files exist
for sp in train val test; do
  if [[ ! -f "$SPLIT_DIR/${sp}_list.txt" ]]; then
    echo "ERROR: missing split list: $SPLIT_DIR/${sp}_list.txt"
    exit 1
  fi
done

# Inference helper (runs on a split directory)
run_infer_for_split () {
  local split=$1
  local in_dir="$TMP_IMG_DIR/$split"   # contains caseID_0000.mha
  local out_dir="$INFER_OUT/$split"
  mkdir -p "$out_dir"

  if [[ ! -d "$in_dir" ]]; then
    echo "[WARN] input dir not found: $in_dir"
    return
  fi

  echo "------ [Infer $split] ------"
  nnUNetv2_predict \
    -i "$in_dir" \
    -o "$out_dir" \
    -d $DATASET_ID \
    -c $CFG \
    -p $PLANS \
    -tr $TRAINER \
    -f $FOLD \
    -chk checkpoint_best.pth

  # Normalize output filenames to caseID.nii.gz according to the split list
  while IFS= read -r cid; do
    [[ -z "$cid" ]] && continue
    if [[ ! -f "$out_dir/$cid.nii.gz" ]]; then
      found="$(ls "$out_dir" | grep -E "^${cid}.*\.nii\.gz$" || true)"
      if [[ -n "$found" ]]; then
        mv "$out_dir/$found" "$out_dir/$cid.nii.gz"
      fi
    fi
  done < "$SPLIT_DIR/${split}_list.txt"
}

run_infer_for_split train
run_infer_for_split val
run_infer_for_split test

echo "Inference completed. Raw predictions are in: $INFER_OUT"
echo "Next step: python 3-runcollect.py"
