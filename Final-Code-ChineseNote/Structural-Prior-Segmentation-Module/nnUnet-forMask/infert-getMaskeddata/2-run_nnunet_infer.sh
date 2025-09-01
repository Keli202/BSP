#!/bin/bash
#SBATCH --job-name=nnunet_infer_all_2d
#SBATCH --partition=gpu
#SBATCH --account=COMS031144
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=nnunet_infer_all_2d_%j.out
#SBATCH --error=nnunet_infer_all_2d_%j.err

echo "========== SLURM job started =========="
echo "SLURM script: $(realpath $0)"
echo "Submit dir: $(pwd)"
echo "Date: $(date)"
echo "========================================"

# ===== 环境 =====
. /user/home/ad21083/initConda.sh
conda activate Detect-BSPnnUnet

set -eo pipefail
shopt -s nullglob

echo "Using GPU(s):"
nvidia-smi --query-gpu=name --format=csv

# ===== nnU-Net 参数（与训练一致，2D & fold0）=====
DATASET_ID=300
CFG="2d"
PLANS="nnUNetResEncUNetLPlans"
TRAINER="nnUNetTrainer"
FOLD=0

# ===== 目录（SixthVision 项目内）=====
BASE_NNUNET="/user/work/ad21083/Detection-BSP/Code/SixthVision-Code/nnUnet"
DATA_DIR="$BASE_NNUNET/data"

SPLIT_DIR="$DATA_DIR/splits_from_npz"                 # train_list.txt / val_list.txt / test_list.txt
TMP_IMG_DIR="$DATA_DIR/tmp_for_infer/images"          # 准备好的 *_0000.mha
INFER_OUT="$DATA_DIR/infer_results"                   # 推理输出（.nii.gz）
mkdir -p "$INFER_OUT"/{train,val,test}

# 权重
export nnUNet_results="$BASE_NNUNET/nnUNet_results"
CKPT="$nnUNet_results/Dataset300_FetalAbdomen/${TRAINER}__${PLANS}__${CFG}/fold_${FOLD}/checkpoint_best.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi
echo "[INFO] Using checkpoint: $CKPT"

# splits 必须在（虽然不强依赖，但用于命名统一&核对）
for sp in train val test; do
  if [[ ! -f "$SPLIT_DIR/${sp}_list.txt" ]]; then
    echo "ERROR: missing split list: $SPLIT_DIR/${sp}_list.txt"
    exit 1
  fi
done

# ===== 推理函数：对整个目录推理 =====
run_infer_for_split () {
  local split=$1
  local in_dir="$TMP_IMG_DIR/$split"   # 包含 case_0000.mha
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

  # 统一命名：确保输出名为 caseID.nii.gz
  while IFS= read -r cid; do
    [[ -z "$cid" ]] && continue
    # 若没有标准名，尝试把匹配的文件改名
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

echo "✅ Inference done. Raw predictions are in: $INFER_OUT"
echo "👉 Next step: python collect_results.py"
