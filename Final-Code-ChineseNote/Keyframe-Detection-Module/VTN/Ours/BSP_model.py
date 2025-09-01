# BSP_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# 基础模块
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: (B, T, C)
        return x + self.pe[:, :x.size(1)].to(x.device)


class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.last_weights = None

    def forward(self, x):  # (B, T, C)
        b, t, c = x.size()
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)
        y = self.fc(y).view(b, 1, c)
        self.last_weights = y.detach().cpu().numpy()
        return x * y

    def get_last_weights(self):
        return self.last_weights


class FusionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.mlp(x)


# ==========================
# 你原来的双向 Transformer（保留）
# ==========================
class BiVideoTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
        num_layers=4,
        use_conv=True,
        conv_window=4,
        causal_conv=True,
        manual_conv_weights=None,
        use_bitrans=True,
        use_se=False,
        se_reduction=16
    ):
        super().__init__()
        self.use_conv = use_conv
        self.causal_conv = causal_conv
        self.conv_window = conv_window
        self.use_bitrans = use_bitrans

        if use_conv:
            padding = 0 if causal_conv else conv_window // 2
            self.conv1d = nn.Conv1d(
                in_channels=feature_dim, out_channels=feature_dim,
                kernel_size=conv_window, padding=padding, groups=feature_dim, bias=False
            )
            with torch.no_grad():
                self.conv1d.weight.zero_()
                if manual_conv_weights is not None:
                    w = torch.tensor(manual_conv_weights, dtype=torch.float32)
                    assert len(w) == conv_window
                    for i in range(feature_dim):
                        self.conv1d.weight[i, 0, :] = w
                else:
                    center = -1 if causal_conv else conv_window // 2
                    self.conv1d.weight[:, :, center] = 1.0

        self.linear_proj = nn.Linear(feature_dim, hidden_dim)
        self.se = SEBlock1D(hidden_dim, reduction=se_reduction) if use_se else None
        self.pos_encoder = PositionalEncoding(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        if use_bitrans:
            self.trans_f = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.trans_b = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.classifier = nn.Linear(hidden_dim * 2, 1)
        else:
            self.trans = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, T, D)
        x = x.to(next(self.parameters()).device)
        if self.use_conv:
            B, T, D = x.shape
            x_conv = x.permute(0, 2, 1)
            if self.causal_conv:
                pad_len = self.conv_window - 1
                x_conv = F.pad(x_conv, (pad_len, 0))
            x = self.conv1d(x_conv).permute(0, 2, 1)

        x = self.linear_proj(x)
        if self.se is not None:
            x = self.se(x)
        x = self.pos_encoder(x)

        if hasattr(self, "trans_f"):
            out_f = self.trans_f(x)
            out_b = self.trans_b(torch.flip(x, [1]))
            out_b = torch.flip(out_b, [1])
            out = torch.cat([out_f, out_b], dim=-1)
        else:
            out = self.trans(x)
        logits = self.classifier(out).squeeze(-1)
        return logits, out

    def get_se_weights(self):
        return self.se.get_last_weights() if self.se is not None else None


# ==========================
# 新：FPN + SMFE（多帧融合 + SE）+ 每阶段双向Transformer
# ==========================
class MultiLayerFusionTransformerFPN(nn.Module):
    """
    结构（自顶向下：11 -> 9 -> 7 -> 5）：
      Stage1:  b11 --SMFE--> base1 --PE--> BiTrans1 -> h1 -> head1
      Stage2:  b9  --SMFE--> low2;  fuse( reduce(h1), low2 ) -> BiTrans2 -> h2 -> head2
      Stage3:  b7  --SMFE--> low3;  fuse( reduce(h2), low3 ) -> BiTrans3 -> h3 -> head3
      Stage4:  b5  --SMFE--> low4;  fuse( reduce(h3), low4 ) -> BiTrans4 -> h4 -> head4
    训练时用四个 head 的融合 MLP 作为最终 logit（你的训练脚本已实现），
    对比损失使用 h4（可 L2 归一化后再算），Gumbel-softmax 辅助与时间平滑保持不变。
    """
    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        nhead=8,
        num_ff=512,
        dropout=0.1,
        layers_per_stage=1,
        use_conv1d=True,
        conv_window=4,
        causal_conv=True,
        manual_conv_weights=None,
        use_se=True,
        se_reduction=16,
        use_bitrans=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_bitrans = use_bitrans
        self.use_conv1d = use_conv1d
        self.causal_conv = causal_conv
        self.conv_window = conv_window

        # 共享的“多帧融合” depthwise Conv1d（SMFE的MF部分）
        if use_conv1d:
            padding = 0 if causal_conv else conv_window // 2
            self.conv1d = nn.Conv1d(
                in_channels=feature_dim, out_channels=feature_dim,
                kernel_size=conv_window, padding=padding, groups=feature_dim, bias=False
            )
            with torch.no_grad():
                self.conv1d.weight.zero_()
                if manual_conv_weights is not None:
                    w = torch.tensor(manual_conv_weights, dtype=torch.float32)
                    assert len(w) == conv_window
                    for i in range(feature_dim):
                        self.conv1d.weight[i, 0, :] = w
                else:
                    center = -1 if causal_conv else conv_window // 2
                    self.conv1d.weight[:, :, center] = 1.0
        else:
            self.conv1d = None

        # 每层各自的 Linear 投影（SMFE的“P”部分）+ 各自的 SE
        self.proj_11 = nn.Linear(feature_dim, hidden_dim)
        self.proj_09 = nn.Linear(feature_dim, hidden_dim)
        self.proj_07 = nn.Linear(feature_dim, hidden_dim)
        self.proj_05 = nn.Linear(feature_dim, hidden_dim)

        self.se1 = SEBlock1D(hidden_dim, se_reduction) if use_se else None
        self.se2 = SEBlock1D(hidden_dim, se_reduction) if use_se else None
        self.se3 = SEBlock1D(hidden_dim, se_reduction) if use_se else None
        self.se4 = SEBlock1D(hidden_dim, se_reduction) if use_se else None

        # 位置编码：仅第一阶段
        self.pos = PositionalEncoding(hidden_dim)

        # 每阶段 Encoder（支持双向）
        def enc_stack():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=nhead, dim_feedforward=num_ff, dropout=dropout, batch_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=layers_per_stage)

        if use_bitrans:
            self.f_enc1 = enc_stack(); self.b_enc1 = enc_stack()
            self.f_enc2 = enc_stack(); self.b_enc2 = enc_stack()
            self.f_enc3 = enc_stack(); self.b_enc3 = enc_stack()
            self.f_enc4 = enc_stack(); self.b_enc4 = enc_stack()
            head_in_dim = hidden_dim * 2
        else:
            self.enc1 = enc_stack()
            self.enc2 = enc_stack()
            self.enc3 = enc_stack()
            self.enc4 = enc_stack()
            head_in_dim = hidden_dim

        # 将上一阶段输出 h*（H 或 2H）降到 H，再与本层 SMFE 输出拼接
        self.reduce_h = nn.Linear(head_in_dim, hidden_dim)

        # 三次融合 MLP：concat(H, H) -> H
        self.fuse1 = FusionMLP(hidden_dim * 2, hidden_dim)  # 用于 Stage2
        self.fuse2 = FusionMLP(hidden_dim * 2, hidden_dim)  # 用于 Stage3
        self.fuse3 = FusionMLP(hidden_dim * 2, hidden_dim)  # 用于 Stage4

        # 四个阶段的 head
        self.head1 = nn.Linear(head_in_dim, 1)
        self.head2 = nn.Linear(head_in_dim, 1)
        self.head3 = nn.Linear(head_in_dim, 1)
        self.head4 = nn.Linear(head_in_dim, 1)

    # --------- 工具函数 ---------
    @staticmethod
    def _slice_levels(x4d):
        # x4d: (B, T, 9, C) -> 取 [5,7,9,11] 对应 [2,4,6,8]
        b5  = x4d[:, :, 2, :]
        b7  = x4d[:, :, 4, :]
        b9  = x4d[:, :, 6, :]
        b11 = x4d[:, :, 8, :]
        return b5, b7, b9, b11

    def _temporal_fuse(self, x):  # 多帧融合（depthwise 1D conv）
        if self.conv1d is None:
            return x
        x_ = x.permute(0, 2, 1)  # (B, C, T)
        if self.causal_conv:
            pad_len = self.conv_window - 1
            x_ = F.pad(x_, (pad_len, 0))
        x_ = self.conv1d(x_)
        return x_.permute(0, 2, 1)

    def _smfe(self, x, proj, se_block):  # SMFE: 多帧融合 + 投影 + SE
        x = self._temporal_fuse(x)      # (B, T, Cin)
        x = proj(x)                     # (B, T, H)
        if se_block is not None:
            x = se_block(x)             # (B, T, H)
        return x

    def _run_bi_encoder(self, base, f_enc, b_enc):  # base: (B, T, H) -> (B, T, 2H)
        out_f = f_enc(base)
        out_b = b_enc(torch.flip(base, [1]))
        out_b = torch.flip(out_b, [1])
        return torch.cat([out_f, out_b], dim=-1)

    # --------- 前向 ---------
    def forward(self, x):
        """
        x: (B, T, 9, C) 或 (B, T, 4, C)（顺序必须是 [5,7,9,11]）
        返回:
          heads: (B, T, 4)
          h4:    (B, T, head_in_dim)  # 供对比损失/评估使用
        """
        if x.dim() != 4:
            raise ValueError(f"Expect x as (B,T,L,C), got {x.shape}")
        B, T, L, C = x.shape

        if L == 9:
            b5, b7, b9, b11 = self._slice_levels(x)
        elif L == 4:
            b5, b7, b9, b11 = x[:, :, 0, :], x[:, :, 1, :], x[:, :, 2, :], x[:, :, 3, :]
        else:
            raise ValueError(f"Expect 4 or 9 levels, got {L}")

        # 先各自做一遍 SMFE（多帧融合 + 投影 + SE）
        s11 = self._smfe(b11, self.proj_11, self.se1)  # (B,T,H)
        s09 = self._smfe(b9,  self.proj_09, self.se2)  # (B,T,H)
        s07 = self._smfe(b7,  self.proj_07, self.se3)  # (B,T,H)
        s05 = self._smfe(b5,  self.proj_05, self.se4)  # (B,T,H)

        # Stage 1: s11 (+ PE) -> BiTrans1
        base1 = self.pos(s11)
        if self.use_bitrans:
            h1 = self._run_bi_encoder(base1, self.f_enc1, self.b_enc1)  # (B,T,2H)
        else:
            h1 = self.enc1(base1)                                       # (B,T,H)
        head1 = self.head1(h1).squeeze(-1)

        # Stage 2: fuse( reduce(h1), s09 ) -> BiTrans2
        h1_red = self.reduce_h(h1)                       # (B,T,H)
        base2 = self.fuse1(torch.cat([h1_red, s09], dim=-1))  # (B,T,H)
        if self.use_bitrans:
            h2 = self._run_bi_encoder(base2, self.f_enc2, self.b_enc2)  # (B,T,2H)
        else:
            h2 = self.enc2(base2)                                       # (B,T,H)
        head2 = self.head2(h2).squeeze(-1)

        # Stage 3: fuse( reduce(h2), s07 ) -> BiTrans3
        h2_red = self.reduce_h(h2)                       # (B,T,H)
        base3 = self.fuse2(torch.cat([h2_red, s07], dim=-1))
        if self.use_bitrans:
            h3 = self._run_bi_encoder(base3, self.f_enc3, self.b_enc3)
        else:
            h3 = self.enc3(base3)
        head3 = self.head3(h3).squeeze(-1)

        # Stage 4: fuse( reduce(h3), s05 ) -> BiTrans4
        h3_red = self.reduce_h(h3)
        base4 = self.fuse3(torch.cat([h3_red, s05], dim=-1))
        if self.use_bitrans:
            h4 = self._run_bi_encoder(base4, self.f_enc4, self.b_enc4)
        else:
            h4 = self.enc4(base4)
        head4 = self.head4(h4).squeeze(-1)

        heads = torch.stack([head1, head2, head3, head4], dim=-1)  # (B, T, 4)
        return heads, h4
