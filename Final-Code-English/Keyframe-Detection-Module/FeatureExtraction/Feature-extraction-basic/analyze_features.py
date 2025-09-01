#!/usr/bin/env python3
"""
Purpose
-------
Sanity-check extracted features with simple KNN and visualizations.
Reads:
    data/processed_features/features/train/*.npz
    data/processed_features/features/test/*.npz
Writes:
    Final-Code-Eng/Keyframe-Detection-Module/FeatureExtraction/Feature-extraction-basic/feature_analysis_output/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from pathlib import Path

# ----------- paths -----------
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

TRAIN_DIR = str(DATA_ROOT / "processed_features" / "features" / "train")
TEST_DIR  = str(DATA_ROOT / "processed_features" / "features" / "test")
SAVE_DIR  = str(Path(__file__).resolve().parent / "feature_analysis_output")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_features_labels(feature_dir):
    all_features, all_labels = [], []
    for file in sorted(os.listdir(feature_dir)):
        if not file.endswith('.npz'):
            continue
        data = np.load(os.path.join(feature_dir, file))
        feats, labels = data['features'], data['labels']
        all_features.append(feats)
        all_labels.append(labels)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels

# load train/test
train_features, train_labels = load_features_labels(TRAIN_DIR)
test_features, test_labels   = load_features_labels(TEST_DIR)

print(f"[Train] Features: {train_features.shape}, Labels: {train_labels.shape}")
print(f"[Test ] Features: {test_features.shape}, Labels: {test_labels.shape}")

# ----------- 2. KNN self-classification (fit on test, predict test) -----------
knn_self = KNeighborsClassifier(n_neighbors=5)
knn_self.fit(test_features, test_labels)
pred_self = knn_self.predict(test_features)
acc_self = accuracy_score(test_labels, pred_self)
print(f"\n[Test self-classification] KNN accuracy: {acc_self:.4f}")
report_self = classification_report(test_labels, pred_self, target_names=["Background", "Keyframe"])
cm_self = confusion_matrix(test_labels, pred_self)
print(report_self)
print(cm_self)
with open(os.path.join(SAVE_DIR, "knn_self_classification_report.txt"), "w") as f:
    f.write("[Test self-classification] KNN accuracy: %.4f\n" % acc_self)
    f.write(report_self)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm_self))

# ----------- 3. KNN train->test -----------
knn_split = KNeighborsClassifier(n_neighbors=5)
knn_split.fit(train_features, train_labels)
pred_split = knn_split.predict(test_features)
acc_split = accuracy_score(test_labels, pred_split)
print(f"\n[Train->Test] KNN accuracy: {acc_split:.4f}")
report_split = classification_report(test_labels, pred_split, target_names=["Background", "Keyframe"])
cm_split = confusion_matrix(test_labels, pred_split)
print(report_split)
print(cm_split)
with open(os.path.join(SAVE_DIR, "knn_train_test_classification_report.txt"), "w") as f:
    f.write("[Train->Test] KNN accuracy: %.4f\n" % acc_split)
    f.write(report_split)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm_split))

# ----------- 4. t-SNE (test) -----------
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    feats_2d = tsne.fit_transform(test_features)
    plt.figure(figsize=(7, 6))
    plt.scatter(feats_2d[test_labels==0, 0], feats_2d[test_labels==0, 1], s=7, c='gray', label='Background', alpha=0.5)
    plt.scatter(feats_2d[test_labels==1, 0], feats_2d[test_labels==1, 1], s=16, c='red', label='Keyframe', alpha=0.7)
    plt.legend()
    plt.title('t-SNE visualization of test features')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'tsne_test_features.png'))
    plt.close()
except Exception as e:
    print(f"t-SNE failed: {e}")

# ----------- 5. PCA (test) -----------
try:
    pca = PCA(n_components=2)
    feats_pca = pca.fit_transform(test_features)
    plt.figure(figsize=(7, 6))
    plt.scatter(feats_pca[test_labels==0, 0], feats_pca[test_labels==0, 1], s=7, c='gray', label='Background', alpha=0.5)
    plt.scatter(feats_pca[test_labels==1, 0], feats_pca[test_labels==1, 1], s=16, c='blue', label='Keyframe', alpha=0.7)
    plt.legend()
    plt.title('PCA visualization of test features')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'pca_test_features.png'))
    plt.close()
except Exception as e:
    print(f"PCA failed: {e}")

# ----------- 6. Heatmap (first 30 dims, test) -----------
try:
    df = pd.DataFrame(test_features[:, :30])
    df['label'] = test_labels
    means = df.groupby('label').mean().values
    plt.figure(figsize=(12, 4))
    sns.heatmap(means, annot=False, cmap="RdBu_r", yticklabels=["Background", "Keyframe"])
    plt.xlabel("Feature Dim (first 30)")
    plt.title("Feature Mean by Class (test set, first 30 dims)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'feature_mean_heatmap_test.png'))
    plt.close()
except Exception as e:
    print(f"Heatmap failed: {e}")

print(f"\nAll analysis files are saved in {SAVE_DIR}")
