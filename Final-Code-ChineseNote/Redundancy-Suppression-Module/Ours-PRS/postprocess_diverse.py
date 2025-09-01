import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from scipy.signal import savgol_filter, find_peaks

# ================== 路径配置 ==================
RESULTS_DIR = "../../Keyframe-Detection-Module/VTN/Ours/eval_output/test"
SAVE_DIR    = "eval_output/test_ablation_new"
MASK_DIR    = "/user/work/ad21083/Detection-BSP/Code/Datasets/acouslic-ai-train-set/acouslic-ai-train-set/masks/stacked_fetal_abdomen/"
os.makedirs(SAVE_DIR, exist_ok=True)

PLOT_DIR     = os.path.join(SAVE_DIR, "plots_threshold0p9")
TXT_DIR      = os.path.join(SAVE_DIR, "txt_threshold0p9")      # 详细 TSV
PERCASE_DIR  = os.path.join(SAVE_DIR, "per_case")              # npz
PRED012_DIR  = os.path.join(SAVE_DIR, "pred012_txt")           # ★ 新增：精简逐帧分类 TXT（仅 frame/pred）
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TXT_DIR,  exist_ok=True)
os.makedirs(PERCASE_DIR, exist_ok=True)
os.makedirs(PRED012_DIR, exist_ok=True)  # ★ NEW

# ================== 全局设置 ==================
NEW_BIN_THRESHOLD = 0.9          # 先按 0.9 建候选 → 后续连段/救援/分配1与2
ENSURE_AT_LEAST_ONE_1 = False    # 无1则 WFSS-pred1 跳过
SMALL_SEG_ALL_KEY_LEN = 5 
# ---------- 概率读取（优先 fused；否则多头均值） ----------
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def get_probs_and_features(data):
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

# ---------- 护峰平滑（不移峰、限制下削幅度） ----------
SG_WIN, SG_ORDER = 5, 2
EMA_ALPHA = 0.3
DOWN_CLAMP = 0.05  # 允许向下最多削 0.05（保证 0.95 仍 >= 0.90）

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
    return np.clip(p_thr, 0.0, 1.0), p_mix  # p_thr 用于判阈，p_mix 仅可视化

# ---------- RLE + 合并小缝 ----------
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

# ---------- 峰值救援 ----------
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

# ---------- 段落化 1/2 ----------
BORDER_W_FIXED  = 3
BORDER_RATIO    = 0.15
USE_RATIO_BORDER = True

def label_segments_interior_border(T, segs):
    """
    段落化规则：
      - 若段长 L < SMALL_SEG_ALL_KEY_LEN（默认5），该段全部标为 1（关键帧），不再设置两端为 2；
      - 否则：边界=2，内部=1（保持原逻辑）。
    返回：
      pred: 长度为 T 的 0/1/2 三分类结果
      ones: 预测为 1 的索引数组（用于 WFSS-pred1 等）
    """
    pred = np.zeros(T, dtype=int)  # 0=BG, 1=Key, 2=Sub(边界/冗余)
    ones = []
    for (a, b) in segs:
        L = b - a
        # ★ 新增：小段全部标为1
        if L < SMALL_SEG_ALL_KEY_LEN:
            pred[a:b] = 1
            ones.extend(range(a, b))
            continue

        # —— 原有“边界=2、内部=1”的逻辑 —— #
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


# ---------- WFSS（均在“筛选后结果”上） ----------
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

# ---------- 读取 GT（★ 三分类：0/1/2） ----------
def get_orig_labels(mask_path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    labels = np.zeros(arr.shape[0], dtype=int)
    for i, frame in enumerate(arr):
        if np.any(frame==1): labels[i]=1
        elif np.any(frame==2): labels[i]=2
        else: labels[i]=0
    return labels

# ================== 主评估 ==================
print("\n====== thr=0.9 | 平滑→阈值连段→峰值救援→段内=1/边=2（基于筛选后的三分类评估） ======\n")

y_true_all, y_pred_all = [], []
wfss_top1_vals, wfss_pred1_vals = [], []
key_cos_list, sub_cos_list = [], []
all_pred_feats, all_gt_feats = [], []

for file in tqdm(sorted(os.listdir(RESULTS_DIR))):
    if not file.endswith('_results.npz'): continue
    case_id = file.replace('_results.npz', '')
    data = np.load(os.path.join(RESULTS_DIR, file))

    # 概率+特征
    probs, features, prob_src = get_probs_and_features(data)

    # 1) 护峰平滑：得到判阈用的 p_thr（以及仅供观察的 p_mix）
    p_thr, p_mix = smooth_probs_guarded(probs)

    # 2) 主通道：阈值→段（去独峰、补小缝）
    segs_main, mask = segments_from_threshold(p_thr, NEW_BIN_THRESHOLD, MIN_RUN_MAIN, MAX_GAP)

    # 3) 峰值救援：允许短而尖的真段（特征护真）
    segs = rescue_short_peaks(probs, p_thr, segs_main, features,
                              thr=NEW_BIN_THRESHOLD, prominence=PK_PROM,
                              max_span=PK_MAX_SPAN, tau_pk=PK_TAU_COS)

    # 4) 段内=1，边=2（这一步得到“筛选后的三分类”）
    pred_multi, selected_idx = label_segments_interior_border(len(probs), segs)

    # 5) 读取 GT（三分类）
    mask_path = os.path.join(MASK_DIR, case_id + '.mha')
    if not os.path.exists(mask_path):
        print(f"[Warn] mask not found for {case_id}, skip."); continue
    gt_multi = get_orig_labels(mask_path)
    if len(gt_multi) != len(pred_multi):
        print(f"[Warn] length mismatch for {case_id}: gt={len(gt_multi)} pred={len(pred_multi)}; skip."); continue

    # 5.1 保存逐case（详细 TSV + npz + 精简 txt）
    # 详细 TSV（含 gt）
    pd.DataFrame({
        "frame": np.arange(len(probs), dtype=int),
        "prob_raw": probs.astype(float),
        "prob_smooth": p_mix.astype(float),
        "prob_thr": p_thr.astype(float),
        "pred": pred_multi.astype(int),     # ★ 你要看的三分类
        "gt": gt_multi.astype(int)          # GT 三分类
    }).to_csv(os.path.join(TXT_DIR, f"{case_id}.txt"), sep="\t", index=False)
    # npz（保留段信息，便于复现）
    np.savez(
        os.path.join(PERCASE_DIR, f"{case_id}_postfilter.npz"),
        pred_multi=pred_multi.astype(np.int16),
        probs_raw=probs.astype(np.float32),
        prob_thr=p_thr.astype(np.float32),
        prob_smooth=p_mix.astype(np.float32),
        segments=np.array(segs, dtype=np.int32)
    )
    # ★ 精简版 TXT：仅逐帧三分类（两列：frame\tpred012），便于快速查看
    np.savetxt(
        os.path.join(PRED012_DIR, f"{case_id}_pred012.txt"),
        np.stack([np.arange(len(pred_multi)), pred_multi], axis=1).astype(int),
        fmt="%d",
        delimiter="\t",
        header="frame\tpred012",
        comments=""
    )

    # 6) 累计评估（都在筛选后三分类上）
    y_true_all.append(gt_multi); y_pred_all.append(pred_multi)
    wt = wfss_top1(selected_idx, probs, gt_multi);   wp = wfss_pred1(pred_multi, gt_multi)
    if wt is not None: wfss_top1_vals.append(wt)
    if wp is not None: wfss_pred1_vals.append(wp)

    # （可选）余弦统计
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

    # 7) 可视化（保留）
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

# ================== 汇总与保存（核心指标） ==================
if y_true_all:
    y_true_all = np.concatenate(y_true_all, 0)
    y_pred_all = np.concatenate(y_pred_all, 0)

    # 三分类报告（可读参考，不影响主表）
    print("\n--- 3-class classification report (frame-level, all test frames) ---")
    print(classification_report(y_true_all, y_pred_all,
                                labels=[0,1,2],
                                target_names=['BG(0)','Key(1)','Sub(2)'],
                                digits=4, zero_division=0))
    print("Confusion Matrix (labels 0,1,2):\n", confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2]))

    # ★ 论文主表要的三个分类指标（把 GT==1 当正类来度量）
    gt_bin = (y_true_all == 1).astype(int)
    pd_bin = (y_pred_all == 1).astype(int)
    prec1 = precision_score(gt_bin, pd_bin, zero_division=0)
    rec1  = recall_score(gt_bin, pd_bin, zero_division=0)
    f1_1  = f1_score(gt_bin, pd_bin, zero_division=0)
else:
    prec1 = rec1 = f1_1 = 0.0
    print("\n[Warn] No frames aggregated for PR/F1.")

# 两个 WFSS（单值，跨 case 均值；pred1只统计有 Key 预测的 case）
wfss_top1_mean  = float(np.mean(wfss_top1_vals))  if len(wfss_top1_vals)>0  else 0.0
wfss_pred1_mean = float(np.mean(wfss_pred1_vals)) if len(wfss_pred1_vals)>0 else 0.0

print("\n--- WFSS (per-case avg; skipping cases with no pred-1) ---")
print(f"WFSS-top1  : {wfss_top1_mean:.4f} | counted: {len(wfss_top1_vals)}")
print(f"WFSS-pred1 : {wfss_pred1_mean:.4f} | counted: {len(wfss_pred1_vals)}")

# 单独保存“五个指标”（主表）
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

# （可选）额外：余弦统计汇总
mean_key_cos = float(np.mean(key_cos_list)) if key_cos_list else 0.0
mean_sub_cos = float(np.mean(sub_cos_list)) if sub_cos_list else 0.0
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
print(" - 主表五指标  :", os.path.join(SAVE_DIR, "metrics_five.csv"))
print(" - 详细逐帧TSV :", TXT_DIR, "(每case一个 .txt)")
print(" - 精简逐帧TXT :", PRED012_DIR, "(每case一个 *_pred012.txt)")
print(" - 段/概率NPZ  :", PERCASE_DIR)
print("\nDone.")
