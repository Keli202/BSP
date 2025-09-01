#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复制 Datasets/processed_features/features/{train,val,test} 到一个新目录，
并把 .npz 里的 labels 替换为【原始 GT masks】生成的帧级标签(>0→1)。
不修改原数据集，其它字段(如 features)保持不变。

长度对齐策略（已启用，且会明确记录是否使用）：
- 若 T_feat != T_gt，则仅对 labels 做“时间轴最近邻重采样”到 T_feat；
- 每个使用了对齐的样本，都会打印 [OK-ALIGNED] 并写入 aligned_cases.txt；
- 未使用对齐（T_feat == T_gt）则打印 [OK]。

直接运行：python fix_features_labels_to_gt_make_new.py
"""

import os
import sys
import csv
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ========== 固定路径配置（按需修改）==========
INPUT_FEATURE_ROOT  = "../FeatureExtraction/Feature-extraction-FPN/Datasets/processed_features/features"          # 原 features 根目录（含 train/val/test）
OUTPUT_FEATURE_ROOT = "Datasets/processed_features_gtlabels/features" # 新的数据集根目录（将写入到这里）
GT_MASK_DIR = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen"
SPLITS = ["train", "val", "test"]

# 期望成功写入的样本总数（确保全部样本都处理到；若你的总数不是 300，请改这里）
EXPECTED_OK = 300

# 报告文件（写到 OUTPUT_FEATURE_ROOT 同级目录）
OUT_DIR_FOR_REPORTS = os.path.dirname(OUTPUT_FEATURE_ROOT.rstrip("/")) or "."
SAVE_REPORT_PATH   = os.path.join(OUT_DIR_FOR_REPORTS, "fix_report.csv")
SAVE_PROBLEM_LIST  = os.path.join(OUT_DIR_FOR_REPORTS, "problem_files.txt")
SAVE_ALIGNED_LIST  = os.path.join(OUT_DIR_FOR_REPORTS, "aligned_cases.txt")

# ========== 工具 ==========
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_gt_labels(mask_path: str) -> np.ndarray:
    """从 GT mask(.mha) 生成帧级标签：该帧存在任意前景(>0)则记 1，否则 0。"""
    itk = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(itk).astype(np.float32)  # (T,H,W)
    T = arr.shape[0]
    return (arr.reshape(T, -1) > 0).any(axis=1).astype(np.int32)

def align_labels_to_len(labels_gt: np.ndarray, T_target: int) -> np.ndarray:
    """
    把长度为 T_gt 的标签，用时间轴最近邻重采样到 T_target。
    保证返回长度 == T_target；不改变除长度外的任何东西。
    """
    T_gt = len(labels_gt)
    if T_gt == T_target:
        return labels_gt.astype(np.int32)
    # 最近邻索引：把 [0..T_target-1] 映射到 [0..T_gt-1]
    idx = np.round(np.linspace(0, T_gt - 1, T_target)).astype(np.int32)
    idx = np.clip(idx, 0, T_gt - 1)
    return labels_gt[idx].astype(np.int32)

def process_one_file(in_npz_path: str, out_npz_path: str, aligned_fh):
    """
    读取输入 .npz（features + 旧 labels），从 GT 生成新 labels，
    与 features 同长度（必要时重采样 labels），写入到 out_npz_path（新目录）。
    返回 (status, record)
      status: 'ok' | 'ok_aligned' | 'missing_gt' | 'error'
    """
    rec = {
        "file": os.path.basename(in_npz_path),
        "status": "",
        "reason": "",
        "T_feat": -1,
        "T_gt": -1,
        "pos_old": -1,
        "pos_new": -1,
    }

    try:
        data = np.load(in_npz_path)
        if "features" not in data.files:
            rec["status"] = "error"; rec["reason"] = "no 'features' in npz"
            print(f"[ERROR] {in_npz_path} -> {rec['reason']}")
            return rec["status"], rec

        feats = data["features"]  # (T, 9, 768) or (T, L, C)
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

        # 与 features 对齐（只对 labels 做最近邻采样）
        labels_new = align_labels_to_len(labels_gt, T_feat)
        rec["pos_new"] = int(np.sum(labels_new))

        status = "ok" if T_feat == T_gt else "ok_aligned"

        # 写新文件：features 原样；labels 用新值；其它字段保留
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

        # 可选：提示正样本帧数变化
        if labels_old is not None and rec["pos_old"] != rec["pos_new"]:
            print(f"        pos_frames: {rec['pos_old']} -> {rec['pos_new']}")

        return rec["status"], rec

    except Exception as e:
        rec["status"] = "error"; rec["reason"] = repr(e)
        print(f"[ERROR] {in_npz_path} -> {rec['reason']}")
        return rec["status"], rec

def main():
    # 基本检查
    if not os.path.isdir(INPUT_FEATURE_ROOT):
        print(f"[FATAL] INPUT_FEATURE_ROOT not found: {INPUT_FEATURE_ROOT}")
        sys.exit(2)
    if not os.path.isdir(GT_MASK_DIR):
        print(f"[FATAL] GT_MASK_DIR not found: {GT_MASK_DIR}")
        sys.exit(2)

    # 创建输出根目录
    ensure_dir(OUTPUT_FEATURE_ROOT)

    # 打开对齐记录文件
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

    # 汇总
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

    # 报告文件
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

    # 强校验：必须达到 EXPECTED_OK
    if EXPECTED_OK is not None:
        if ok_count != EXPECTED_OK:
            print(f"[FAIL] OK count {ok_count} != EXPECTED_OK {EXPECTED_OK}")
            sys.exit(1)

    # 若存在缺失/错误，也判失败
    if problems:
        print(f"[FAIL] Found {len(problems)} problem files (missing_gt/error).")
        sys.exit(1)

    print("[DONE] All good.")
    sys.exit(0)

if __name__ == "__main__":
    main()
