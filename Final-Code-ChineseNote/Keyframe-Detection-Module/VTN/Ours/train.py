import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from BSP_model import MultiLayerFusionTransformerFPN  # 👈 用FPN模型
import matplotlib.pyplot as plt
import random

# ========== 可复现设置 ==========
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# 如需完全确定性，可打开下面两行（会稍慢）：
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# -------- 配置 --------
TRAIN_DIR = "Datasets/processed_features_gtlabels/features/train"
VAL_DIR   = "Datasets/processed_features_gtlabels/features/val"
SAVE_DIR  = "checkpoints"
EPOCHS = 150
BATCH_SIZE = 1  # 可变长度视频，建议1；>1需自定义collate+mask
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 20
GRAD_CLIP_NORM = 1.0  # 防止偶发不稳定，若不需要可设为 None

# ====== 损失项权重（建议值） ======
LAMBDA_CONTRASTIVE = 0.07   # 对比损失
LAMBDA_GS = 0.30            # soft(Gumbel) 辅助项
LAMBDA_TMP = 0.10           # 时间平滑（相邻帧差），若不想用设 0.0
AUX_WEIGHT = 0.05           # 四个阶段 head 的 deep supervision 权重

# ====== Gumbel-Softmax 温度退火（建议值） ======
TAU_START = 1.5
TAU_END   = 0.5
TAU_GAMMA = 0.98  # 每个epoch乘一次

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert BATCH_SIZE == 1, "当前实现依赖 batch_size=1（可变T）。要>1需要自定义collate与mask。"

# ===== 官方风格的二分类 Gumbel：把1个logit升维为[0, z]再用F.gumbel_softmax =====
def gumbel_sigmoid_official(logits, tau=1.0, hard=False):
    zeros = torch.zeros_like(logits)
    logits2 = torch.stack([zeros, logits], dim=-1)  # (..., 2) -> [背景=0, 关键帧=z]
    probs2 = F.gumbel_softmax(logits2, tau=tau, hard=hard, dim=-1)
    return probs2[..., 1]

# ====== 形状助手：严格但只做“语义明确”的变换 ======
def to_btlc(x, name="feats"):
    """
    接受以下形状并转为 (B,T,L,C):
      - (T, L, C)       -> unsqueeze(0) -> (1,T,L,C)         # 数据集直接返回帧序列
      - (1, T, L, C)    -> 原样返回                           # DataLoader 已加 batch
      - (B, 1, T, L, C) -> squeeze(1)   -> (B,T,L,C)         # 你报错里的情况
    其他维度直接报错（避免误操作）。
    """
    if x.dim() == 3:
        # (T,L,C)
        out = x.unsqueeze(0)
        return out
    elif x.dim() == 4:
        # (B,T,L,C)
        return x
    elif x.dim() == 5:
        # 只接受第二维为1的“多余壳子”
        if x.shape[1] != 1:
            raise ValueError(f"{name}: unexpected 5D shape {tuple(x.shape)}; second dim must be 1 to squeeze safely.")
        out = x.squeeze(1)
        return out
    else:
        raise ValueError(f"{name}: unsupported shape {tuple(x.shape)}; expect (T,L,C)|(1,T,L,C)|(B,1,T,L,C).")

# ===== 融合 head 的小 MLP（把四个阶段的logits融合成主logit）=====
class HeadFusionMLP(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):  # x: (B, T, 4)
        return self.mlp(x).squeeze(-1)  # (B, T)

class VideoFeatureDataset(Dataset):
    def __init__(self, folder_path):
        self.features = []
        self.labels = []
        for file in sorted(os.listdir(folder_path)):
            if not file.endswith('.npz'):
                continue
            data = np.load(os.path.join(folder_path, file))
            # features 可能是 (T, 9, 768) 或 (T, 4, 768)
            self.features.append(torch.tensor(data['features'], dtype=torch.float32))
            self.labels.append(torch.tensor(data['labels'], dtype=torch.float32))
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def contrastive_loss(features, labels, margin=1.0):
    """
    欧氏距离版对比损失。请先 F.normalize(features) 稳定尺度。
    features: (T, C_fpn) —— 使用 FPN 最终阶段特征 h4
    """
    device = features.device
    T = features.shape[0]
    if T < 2:
        return torch.tensor(0.0, device=device)
    pair_idx1, pair_idx2 = torch.triu_indices(T, T, offset=1)
    f1 = features[pair_idx1]
    f2 = features[pair_idx2]
    l1 = labels[pair_idx1]
    l2 = labels[pair_idx2]
    dist = torch.norm(f1 - f2, dim=1)
    label_eq = (l1 == l2).float()
    loss = label_eq * dist.pow(2) + (1 - label_eq) * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()

# ===== 数据 =====
train_dataset = VideoFeatureDataset(TRAIN_DIR)
val_dataset   = VideoFeatureDataset(VAL_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== 模型（FPN）=====
# 自动读出维度：最后一维是通道768；第二维是层数(4或9)
sample_feats = train_dataset[0][0]
assert sample_feats.dim() == 3, f"期望 (T, L, C)，但拿到 {sample_feats.shape}"
_, L_levels, C_dim = sample_feats.shape
print(f"[INFO] feature shape per frame: levels={L_levels}, dim={C_dim}")

model = MultiLayerFusionTransformerFPN(
    feature_dim=C_dim, hidden_dim=256,
    layers_per_stage=1,             # 每阶段1层
    nhead=8, num_ff=512, dropout=0.1,
    use_conv1d=True,                # ✅ pre-frames 时间卷积
    conv_window=4, causal_conv=True, manual_conv_weights=[0.1, 0.15, 0.25, 0.5],
    use_se=True, se_reduction=16,   # ✅ SE
    use_bitrans=True                # ✅ 每阶段双向
).to(device)

fusion_mlp = HeadFusionMLP(in_dim=4, hidden_dim=16).to(device)

# -------- pos_weight --------
all_labels = []
for _, labels in train_dataset:
    all_labels.extend(labels.cpu().numpy().tolist())
num_pos = int(sum(np.array(all_labels) == 1))
num_neg = int(sum(np.array(all_labels) == 0))
pos_weight_value = (num_neg / num_pos) if num_pos > 0 else 1.0
print(f"Train set: pos={num_pos}, neg={num_neg}, pos_weight={pos_weight_value:.2f}")

pos_weight = torch.tensor([pos_weight_value]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 优化器：同时训练 FPN 和 fusion_mlp
optimizer = torch.optim.Adam(list(model.parameters()) + list(fusion_mlp.parameters()), lr=LEARNING_RATE)

# ===== 日志容器 =====
train_losses, val_losses = [], []
bce_losses, contrastive_losses = [], []
soft_losses, temp_losses, aux_losses, taus = [], [], [], []

best_val_loss = float('inf')
best_model_path = os.path.join(SAVE_DIR, 'best_model_fpn.pth')
no_improve_epochs = 0
tau = TAU_START

# ---- 细节日志CSV ----
loss_log = open(os.path.join(SAVE_DIR, 'loss_detail_log.csv'), 'w')
loss_log.write('epoch,train_total,train_bce,train_contrast,train_soft,train_temp,train_aux,val_total,tau\n')

# ====== 训练 ======
for epoch in range(1, EPOCHS + 1):
    model.train()
    fusion_mlp.train()

    total_train_loss = total_bce_loss = 0.0
    total_contrast_loss = total_soft_loss = 0.0
    total_temp_loss = total_aux_loss = 0.0

    for batch_idx, (feats, labels) in enumerate(train_loader):
        # ---- 形状处理：确保是 (B,T,L,C) ----
        raw_shape = tuple(feats.shape)
        feats = to_btlc(feats, name="feats").to(device)  # (B,T,L,C)
        B, T, L, C = feats.shape

        # labels: 允许 (T,) 或 (1,T)
        if labels.dim() == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)  # (T,)
        elif labels.dim() != 1:
            raise ValueError(f"labels shape {tuple(labels.shape)} not supported; expect (T,) or (1,T).")
        if labels.shape[0] != T:
            raise ValueError(f"labels length {labels.shape[0]} != T {T}")

        labels = labels.to(device)

        # 前向
        heads, h4 = model(feats)  # heads: (B,T,4), h4: (B,T,Cf)
        if heads.dim() != 3 or heads.shape[0] != B or heads.shape[1] != T or heads.shape[2] != 4:
            raise RuntimeError(f"Model heads shape unexpected: got {tuple(heads.shape)}, expect (B,T,4)")
        if h4.dim() != 3 or h4.shape[:2] != (B, T):
            raise RuntimeError(f"Model h4 shape unexpected: got {tuple(h4.shape)}, expect (B,T,Cf)")

        # 只支持 B=1 的训练（当前实现假定）
        if B != 1:
            raise RuntimeError(f"Current training assumes B=1, but got B={B}")
        heads = heads.squeeze(0)  # (T,4)
        h4 = h4.squeeze(0)        # (T,Cf)

        # 主logit（融合四个head）
        fused_logits = fusion_mlp(heads.unsqueeze(0)).squeeze(0)  # (T,)

        # --- 形状打印（仅首个 epoch 的前两个 batch 打印，避免刷屏） ---
        if epoch == 1 and batch_idx < 2:
            print(f"[Shape] raw feats: {raw_shape} -> model input (B,T,L,C)={tuple(feats.shape)}")
            print(f"[Shape] heads (B,T,4) before squeeze: {(1, T, 4)}; after squeeze: {tuple(heads.shape)}")
            print(f"[Shape] h4 (B,T,Cf) before squeeze: {(1, T, h4.shape[-1])}; after squeeze: {tuple(h4.shape)}")
            print(f"[Shape] fused_logits (T,)={tuple(fused_logits.shape)}, labels (T,)={tuple(labels.shape)}")

        # L2 归一化后再做对比损失（用 h4）
        h4 = F.normalize(h4, p=2, dim=1)

        # === 主损失：BCE ===
        loss_bce = criterion(fused_logits, labels)

        # === 对比损失 ===
        loss_contrast = contrastive_loss(h4, labels)

        # === soft(Gumbel) 辅助（对 fused_logits 做） ===
        y_soft = gumbel_sigmoid_official(fused_logits, tau=tau, hard=False)  # (T,)
        loss_soft = F.binary_cross_entropy(y_soft, labels)

        # === 时间平滑（相邻帧差，基于 y_soft） ===
        if y_soft.numel() > 1 and LAMBDA_TMP > 0:
            loss_temp = ((y_soft[1:] - y_soft[:-1])**2).mean()
        else:
            loss_temp = torch.zeros((), device=device)

        # === Deep Supervision：四个head 与 labels 的BCE（小权重） ===
        loss_aux = 0.0
        for k in range(4):
            loss_aux = loss_aux + criterion(heads[:, k], labels)
        loss_aux = AUX_WEIGHT * loss_aux

        # === 总损失 ===
        loss = (
            loss_bce
            + LAMBDA_CONTRASTIVE * loss_contrast
            + LAMBDA_GS * loss_soft
            + LAMBDA_TMP * loss_temp
            + loss_aux
        )

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(fusion_mlp.parameters()), GRAD_CLIP_NORM)
        optimizer.step()

        total_train_loss += loss.item()
        total_bce_loss += loss_bce.item()
        total_contrast_loss += loss_contrast.item()
        total_soft_loss += loss_soft.item()
        total_temp_loss += loss_temp.item()
        total_aux_loss += loss_aux.item()

    # ---- 均值统计 ----
    ntr = len(train_loader)
    avg_train_loss = total_train_loss / ntr
    avg_bce_loss = total_bce_loss / ntr
    avg_contrast_loss = total_contrast_loss / ntr
    avg_soft_loss = total_soft_loss / ntr
    avg_temp_loss = total_temp_loss / ntr
    avg_aux_loss = total_aux_loss / ntr

    train_losses.append(avg_train_loss)
    bce_losses.append(avg_bce_loss)
    contrastive_losses.append(avg_contrast_loss)
    soft_losses.append(avg_soft_loss)
    temp_losses.append(avg_temp_loss)
    aux_losses.append(avg_aux_loss)
    taus.append(tau)

    # -------- Validation（评估阶段不用 Gumbel/时间平滑/AUX，仅看 fused BCE） --------
    model.eval()
    fusion_mlp.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = to_btlc(feats, name="feats(val)").to(device)  # (B,T,L,C)
            B, T, _, _ = feats.shape
            if labels.dim() == 2 and labels.shape[0] == 1:
                labels = labels.squeeze(0)
            elif labels.dim() != 1:
                raise ValueError(f"labels(val) shape {tuple(labels.shape)} not supported; expect (T,) or (1,T).")
            if labels.shape[0] != T:
                raise ValueError(f"labels(val) length {labels.shape[0]} != T {T}")
            labels = labels.to(device)

            heads, _ = model(feats)                  # (B,T,4)
            if B != 1:
                raise RuntimeError(f"Current validation assumes B=1, but got B={B}")
            heads = heads.squeeze(0)                 # (T,4)
            fused_logits = fusion_mlp(heads.unsqueeze(0)).squeeze(0)  # (T,)
            loss_bce = criterion(fused_logits, labels)
            total_val_loss += loss_bce.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # ---- 打印 & 写CSV ----
    print(f"[Epoch {epoch}/{EPOCHS}] Train={avg_train_loss:.4f} | "
          f"BCE={avg_bce_loss:.4f} | Con={avg_contrast_loss:.4f} | "
          f"Soft={avg_soft_loss:.4f} | Temp={avg_temp_loss:.4f} | Aux={avg_aux_loss:.4f} | "
          f"Val(BCE)={avg_val_loss:.4f} | tau={tau:.3f}")

    loss_log.write(
        f"{epoch},{avg_train_loss:.6f},{avg_bce_loss:.6f},"
        f"{avg_contrast_loss:.6f},{avg_soft_loss:.6f},{avg_temp_loss:.6f},"
        f"{avg_aux_loss:.6f},{avg_val_loss:.6f},{tau:.4f}\n"
    )
    loss_log.flush()

    # ---- 保存最佳模型（按 Val BCE）----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        try:
            torch.save({
                'model': model.state_dict(),
                'fusion_mlp': fusion_mlp.state_dict()
            }, best_model_path)
            print(f" Saved best model at {best_model_path}")
        except Exception as e:
            print(f"[Warning] Failed to save model: {e}")
    else:
        no_improve_epochs += 1
        print(f"No improvement in val loss for {no_improve_epochs} epoch(s).")

    if no_improve_epochs >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}. Training terminated.")
        break

    # ---- 温度退火（越训越“硬”）----
    tau = max(TAU_END, tau * TAU_GAMMA)

loss_log.close()

# --- 可视化曲线 ---
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation')
plt.plot(range(1, len(bce_losses) + 1), bce_losses, label='Train BCE')
plt.plot(range(1, len(contrastive_losses) + 1), contrastive_losses, label='Train Contrastive')
plt.plot(range(1, len(soft_losses) + 1), soft_losses, label='Train Soft')
plt.plot(range(1, len(temp_losses) + 1), temp_losses, label='Train Temp')
plt.plot(range(1, len(aux_losses) + 1), aux_losses, label='Train Aux (DS)')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'))

with open(os.path.join(SAVE_DIR, 'train_log.txt'), 'w') as f:
    for i, (t, v) in enumerate(zip(train_losses, val_losses)):
        f.write(f"Epoch {i+1}: Train Loss = {t:.6f}, Val Loss = {v:.6f}\n")
    f.write(f"\nL_CON={LAMBDA_CONTRASTIVE}, L_GS={LAMBDA_GS}, L_TMP={LAMBDA_TMP}, AUX_WEIGHT={AUX_WEIGHT}, "
            f"TAU_START={TAU_START}, TAU_END={TAU_END}, TAU_GAMMA={TAU_GAMMA}, "
            f"GRAD_CLIP_NORM={GRAD_CLIP_NORM}\n")

print("Training finished! See 'loss_detail_log.csv' and 'loss_curve.png' for logs.")
