import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from BSP_model import VideoTransformer
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# ========== 配置 ==========
SPLIT = 'test'
DATA_DIR = f"../FeatureExtraction/Feature-extraction-basic/Datasets/processed_features/features/{SPLIT}"
MODEL_PATH = 'checkpoints/best_model.pth'
SAVE_DIR = f'eval_output/{SPLIT}'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(input_dim):
    model = VideoTransformer(feature_dim=input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def evaluate_sample(npz_path, model):
    data = np.load(npz_path)
    features = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0).to(device)
    true_labels = data['labels']
    with torch.no_grad():
        logits, feats = model(features)  # (1, T, 1)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T,)
        pred_labels = (probs > 0.5).astype(int)
        feats_np = feats.squeeze(0).cpu().numpy()
    return true_labels, pred_labels, probs, feats_np

def compute_cosine_similarity(features, true_labels, pred_labels):
    pred_kf_idx = np.where(pred_labels == 1)[0]
    true_kf_idx = np.where(true_labels == 1)[0]
    if len(true_kf_idx) == 0 or len(pred_kf_idx) == 0:
        return None, None
    pred_kf_feats = features[pred_kf_idx]
    true_kf_feats = features[true_kf_idx]
    S = cosine_similarity(pred_kf_feats, true_kf_feats)
    part1 = S.max(axis=1).sum() / len(pred_kf_feats)
    part2 = S.max(axis=0).sum() / len(true_kf_feats)
    mean_sim = 0.5 * (part1 + part2)
    return mean_sim, S

model = None
all_true, all_pred = [], []
all_mean_sim = []
all_probs = []
case_count = 0
case_with_keyframe = 0

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith('.npz'):
        continue
    npz_path = os.path.join(DATA_DIR, file)
    data = np.load(npz_path)
    features_np = data['features']
    if model is None:
        model = load_model(features_np.shape[-1])
    true_labels, pred_labels, probs, out_features = evaluate_sample(npz_path, model)
    mean_sim, sim_matrix = compute_cosine_similarity(out_features, true_labels, pred_labels)

    all_true.extend(true_labels.tolist())
    all_pred.extend(pred_labels.tolist())
    all_probs.extend(probs.tolist())
    case_count += 1

    base_name = os.path.splitext(file)[0]
    binc_true = np.bincount(true_labels.astype(int), minlength=2)
    binc_pred = np.bincount(pred_labels.astype(int), minlength=2)
    print(f"{file}: True label count: {binc_true}, Pred label count: {binc_pred}")

    # 保存每帧概率到csv文件（老师要求）
    df = pd.DataFrame({
        'frame_idx': np.arange(len(probs)),
        'prob_keyframe': probs,
        'true_label': true_labels,
        'pred_label': pred_labels
    })
    csv_path = os.path.join(SAVE_DIR, f"{base_name}_probs.csv")
    df.to_csv(csv_path, index=False)

    # 可视化与保存流程
    if sim_matrix is not None:
        all_mean_sim.append(mean_sim)
        case_with_keyframe += 1
        plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f"Cosine Similarity Heatmap - {base_name}")
        plt.xlabel("True Keyframes")
        plt.ylabel("Predicted Keyframes")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_cosine_heatmap.png"))
        plt.clf()
        print(f"[Done] {file} - Mean Cosine Similarity: {mean_sim:.4f}")
    else:
        print(f"[Done] {file} - No keyframes for comparison.")

    # 标签对比图
    plt.figure(figsize=(12, 3))
    plt.plot(true_labels, label='True', color='green', linewidth=1)
    plt.plot(pred_labels, label='Predicted', color='orange', linestyle='--', linewidth=1)
    plt.title(f"Label Comparison - {base_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_label_comparison.png"))
    plt.clf()

    # 概率曲线
    plt.figure(figsize=(12, 4))
    plt.plot(probs, label='Prob Keyframe')
    plt.legend()
    plt.title(f"Sigmoid Probability - {base_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{base_name}_probs.png"))
    plt.clf()

    np.savez(os.path.join(SAVE_DIR, f"{base_name}_results.npz"),
             true_labels=true_labels,
             pred_labels=pred_labels,
             probs=probs,
             mean_cosine_similarity=mean_sim,
             features=out_features)

# -------- 统计全局分类指标和平均相似度 --------
target_names = ['Background', 'Keyframe']
print("\n--- True label distribution (all frames):", Counter(all_true))
print("--- Predicted label distribution (all frames):", Counter(all_pred))

try:
    print("\n--- Classification Report (all frames, threshold=0.5) ---")
    print(classification_report(all_true, all_pred, target_names=target_names, digits=4))
    print("\n--- Confusion Matrix (all frames, threshold=0.5) ---")
    print(confusion_matrix(all_true, all_pred))
except Exception as e:
    print(f"Failed to compute classification report or confusion matrix: {e}")

# -------- 输出平均余弦相似度和统计信息 --------
if all_mean_sim:
    print(f"\n--- Mean cosine similarity (cases with keyframes): {np.mean(all_mean_sim):.4f} ---")
    print(f"Evaluated on {case_with_keyframe} / {case_count} cases (cases without keyframes are excluded from mean).")
else:
    print("\nNo cases with keyframes for cosine similarity calculation.")

# --------- 阈值敏感性分析，曲线绘制和阈值0.8 summary ---------
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
precisions, recalls, f1s = [], [], []
all_true_arr = np.array(all_true)
all_probs_arr = np.array(all_probs)

for thresh in thresholds:
    preds = (all_probs_arr > thresh).astype(int)
    precisions.append(precision_score(all_true_arr, preds, zero_division=0))
    recalls.append(recall_score(all_true_arr, preds, zero_division=0))
    f1s.append(f1_score(all_true_arr, preds, zero_division=0))

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, marker='o', label="Precision")
plt.plot(thresholds, recalls, marker='s', label="Recall")
plt.plot(thresholds, f1s, marker='^', label="F1-score")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1-score vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "prf_vs_thresholds.png"))
plt.close()

print("\n========== Precision/Recall/F1 at Different Thresholds ==========")
for t, p, r, f in zip(thresholds, precisions, recalls, f1s):
    print(f"Threshold={t:.2f}: Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

# --------- 输出0.8阈值下的主summary ---------
preds_08 = (all_probs_arr > 0.8).astype(int)
print("\n--- Classification Report (all frames, threshold=0.8) ---")
print(classification_report(all_true_arr, preds_08, target_names=target_names, digits=4))
print("\n--- Confusion Matrix (all frames, threshold=0.8) ---")
print(confusion_matrix(all_true_arr, preds_08))
