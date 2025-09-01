#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ================== 路径与输出 ==================
RESULTS_DIR = "../../Keyframe-Detection-Module/VTN/MMsummary/eval_output/test"   # *_results.npz
SAVE_DIR    = "eval_output/test_ablation_new"
MASK_DIR    = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen/"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "per_case"), exist_ok=True)

# ================== 关键阈值（务必区分！） ==================
# 1) 概率阈值：用来从二分类概率中筛入候选关键帧（≥0.9 才进入候选集）
PROB_THRESHOLD = 0.9

# 2) 相似度阈值：冗余覆盖使用的“余弦相似度”阈值（>0.96 视为冗余）
SIMILARITY_TAU = 0.96

# 其他：每个 case 最多保留多少“关键帧1”
MAX_FRAMES = 100

# ================== 冗余筛选（保持你原逻辑不变） ==================
def key_and_redundant_selection(features, probs, tau, prob_threshold, max_frames):
    """
    参数：
      - tau:             余弦相似度阈值（冗余覆盖用），例如 0.96
      - prob_threshold:  概率阈值（候选筛入用），例如 0.9
    过程：
      1) 按概率降序遍历；
      2) 概率 < prob_threshold 的帧直接跳出（不进候选）；
      3) 未被覆盖的帧 → 记为“关键帧1”；其与其它帧的余弦相似度 > tau 的，都被覆盖成“冗余2”；
      4) 最多选 max_frames 个关键帧1。
    返回：
      selected_idx:   被保留为“关键帧1”的索引集合
      redundant_idx:  被标记为“冗余2”的索引集合
    """
    idx_sorted = np.argsort(-probs)
    selected, used = [], np.zeros(len(probs), dtype=bool)
    redundant = set()
    for idx in idx_sorted:
        # —— 这里用“概率阈值”控制候选筛入 —— #
        if probs[idx] < prob_threshold:
            break
        if used[idx]:
            redundant.add(idx)       # 已被覆盖 → 记为“2”
            continue
        selected.append(idx)         # 新的“关键帧1”
        used[idx] = True
        # —— 这里用“相似度阈值”进行冗余覆盖 —— #
        sims = cosine_similarity(features[idx:idx + 1], features)[0]
        new_redundant = np.where((sims > tau) & (~used))[0]
        redundant.update(new_redundant)
        used = used | (sims > tau)
        if len(selected) >= max_frames:
            break
    return np.array(selected, dtype=int), np.array(list(redundant), dtype=int)

# ================== 读取原始 0/1/2 标签（来自 .mha 掩膜） ==================
def get_orig_labels(mask_path):
    """
    返回长度为 T 的整型数组：0/1/2
      - 含像素值1 → 标签1（Key）
      - 含像素值2 → 标签2（Sub）
      - 均不含     → 标签0（BG）
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

# ================== WFSS 定义（均在“冗余筛选后”的结果上计算） ==================
def wfss_top1(selected_idx, probs, gt_multi):
    """
    WFSS–top1 (official)：
      对每个 case，在“冗余筛选后的关键帧集合 selected_idx”中，
      取概率最高的一帧进行打分：GT=1→1.0，GT=2→0.6，GT=0→0.0。
      再跨 case 求平均。
    """
    if len(selected_idx) == 0:
        return 0.0
    top_local = selected_idx[np.argmax(probs[selected_idx])]
    g = gt_multi[top_local]
    return 1.0 if g == 1 else (0.6 if g == 2 else 0.0)

def wfss_pred1(pred_multi, gt_multi):
    """
    WFSS–pred1 (ours)：
      对每个 case，取“冗余筛选后的关键帧集合 Ω = {t | pred_multi[t]==1}”，
      若 Ω 为空则跳过该 case；
      否则逐帧打分（GT=1→1.0，GT=2→0.6，GT=0→0.0）并取平均，最后跨 case 求均值。
    """
    pos_idx = np.where(pred_multi == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    gs = gt_multi[pos_idx]
    scores = np.where(gs == 1, 1.0, np.where(gs == 2, 0.6, 0.0))
    return float(np.mean(scores))

# ================== 主评估流程 ==================
print("\n====== Core evaluation (3-class metrics + WFSS-top1 + WFSS-pred1) ======\n")

# 帧级累计（用于三分类指标）
y_true_all, y_pred_all = [], []

# WFSS（按 case 分数）
wfss_top1_list, wfss_pred1_list = [], []
wfss_rows = []   # 保存逐 case 的两类 WFSS 与是否纳入 pred1 统计

# （可选）相似度统计
key_cos_list, sub_cos_list = [], []
all_pred_feats, all_gt_feats = [], []

for file in tqdm(sorted(os.listdir(RESULTS_DIR))):
    if not file.endswith('_results.npz'):
        continue
    case_id = file.replace('_results.npz', '')
    path = os.path.join(RESULTS_DIR, file)
    data = np.load(path)

    # 兼容不同键名
    probs    = data['fused_probs'] if 'fused_probs' in data else data['probs']    # 逐帧“关键”概率
    features = data['h4_features'] if 'h4_features' in data else data['features'] # 对应特征 (T,C)

    # —— 先概率筛入（≥0.9），再按相似度冗余覆盖（>0.96） —— #
    selected_idx, redundant_idx = key_and_redundant_selection(
        features, probs, tau=SIMILARITY_TAU, prob_threshold=PROB_THRESHOLD, max_frames=MAX_FRAMES
    )

    # —— 形成三分类预测（0=BG，1=关键帧，2=冗余/次关键帧） —— #
    pred_multi = np.zeros_like(probs, dtype=int)
    pred_multi[redundant_idx] = 2
    pred_multi[selected_idx]  = 1

    # —— 读取 GT（0/1/2）并长度对齐检查 —— #
    mask_path = os.path.join(MASK_DIR, case_id + '.mha')
    if not os.path.exists(mask_path):
        print(f"[Warn] mask not found for {case_id}, skip.")
        continue
    gt_multi = get_orig_labels(mask_path)
    if len(gt_multi) != len(pred_multi):
        print(f"[Warn] length mismatch for {case_id}: gt={len(gt_multi)} pred={len(pred_multi)}; skip.")
        continue

    # —— 保存每个 case 的筛选后三分类与索引（便于复查与可视化） —— #
    np.savez(
        os.path.join(SAVE_DIR, "per_case", f"{case_id}_postselect.npz"),
        pred_multi=pred_multi,
        selected_idx=selected_idx,
        redundant_idx=redundant_idx,
        probs=probs
    )

    # —— 帧级三分类累计 —— #
    y_true_all.append(gt_multi)
    y_pred_all.append(pred_multi)

    # —— WFSS（注意：两者都在“筛选后集合”上定义） —— #
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

    # —— （可选）特征相似度统计 —— #
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

# ================== 汇总与保存 ==================
# —— 帧级三分类指标（只报 Key/Sub 的 P/R/F1 + Acc，满足你的写作习惯） —— #
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

    # 详细报告（如不想用 macro，可忽略报告里的 macro 行）
    with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(
            y_true_all, y_pred_all,
            labels=[0,1,2],
            target_names=['BG(0)','Key(1)','Sub(2)'],
            digits=4, zero_division=0
        ))

    # 混淆矩阵图
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

# —— WFSS 聚合（pred1 只对 Ω≠∅ 的 case 求均值，并报告纳入个数） —— #
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

# —— 可选：特征相似度汇总 —— #
mean_key_cos = float(np.mean(key_cos_list)) if key_cos_list else 0.0
mean_sub_cos = float(np.mean(sub_cos_list)) if sub_cos_list else 0.0
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
