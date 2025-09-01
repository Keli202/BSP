# BSP_model.py
# NOTE: Logic is unchanged. Only lightweight comments and docstrings are added to clarify usage.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# Basic building blocks
# ==========================
class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding. Input: (B, T, C)."""
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
    """
    1D Squeeze-and-Excitation over time. Input/Output: (B, T, C).
    Saves the last channel weights for optional inspection via get_last_weights().
    """
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
    """Small MLP used in FPN fusion stages. Shape preserved across time dimension."""
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
# Original Bi-directional Transformer (kept for completeness)
# Input: (B, T, D) -> logits (B, T), features (B, T, H or 2H)
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

        # Optional depthwise temporal conv (causal by default). Groups=feature_dim keeps channels independent.
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
# Ours: FPN + SMFE + Bi-Transformers (per stage) with 4 heads
# Input features per frame:
#   (T, 9, 768) from ViT blocks -> we internally slice to [5,7,9,11]
#   or (T, 4, 768) if you already kept only [5,7,9,11]
# Forward expects (B, T, L, C), returns:
#   heads: (B, T, 4), h4: (B, T, H or 2H)
# ==========================
class MultiLayerFusionTransformerFPN(nn.Module):
    """
    Top-down stages (11 -> 9 -> 7 -> 5). Each stage:
      - temporal depthwise conv (SMFE) + Linear proj + optional SE
      - (Bi)Transformer encoding
      - per-stage head (logit)
    A small fusion MLP (outside this module) combines the 4 stage logits.
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

        # Shared depthwise temporal conv (SMFE's multi-frame fusion)
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

        # Per-level projection + optional SE
        self.proj_11 = nn.Linear(feature_dim, hidden_dim)
        self.proj_09 = nn.Linear(feature_dim, hidden_dim)
        self.proj_07 = nn.Linear(feature_dim, hidden_dim)
        self.proj_05 = nn.Linear(feature_dim, hidden_dim)

        self.se1 = SEBlock1D(hidden_dim, se_reduction) if use_se else None
        self.se2 = SEBlock1D(hidden_dim, se_reduction) if use_se else None
        self.se3 = SEBlock1D(hidden_dim, se_reduction) if use_se else None
        self.se4 = SEBlock1D(hidden_dim, se_reduction) if use_se else None

        # Positional encoding on the top stage
        self.pos = PositionalEncoding(hidden_dim)

        # Encoder stacks
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

        # Reduce previous stage h* to H before fusing with current SMFE output
        self.reduce_h = nn.Linear(head_in_dim, hidden_dim)

        # Fusion MLPs for stages 2/3/4
        self.fuse1 = FusionMLP(hidden_dim * 2, hidden_dim)
        self.fuse2 = FusionMLP(hidden_dim * 2, hidden_dim)
        self.fuse3 = FusionMLP(hidden_dim * 2, hidden_dim)

        # Stage heads (one logit per stage)
        self.head1 = nn.Linear(head_in_dim, 1)
        self.head2 = nn.Linear(head_in_dim, 1)
        self.head3 = nn.Linear(head_in_dim, 1)
        self.head4 = nn.Linear(head_in_dim, 1)

    # --------- helpers ---------
    @staticmethod
    def _slice_levels(x4d):
        # x4d: (B, T, 9, C) -> keep levels [5,7,9,11] == indices [2,4,6,8]
        b5  = x4d[:, :, 2, :]
        b7  = x4d[:, :, 4, :]
        b9  = x4d[:, :, 6, :]
        b11 = x4d[:, :, 8, :]
        return b5, b7, b9, b11

    def _temporal_fuse(self, x):
        if self.conv1d is None:
            return x
        x_ = x.permute(0, 2, 1)  # (B, C, T)
        if self.causal_conv:
            pad_len = self.conv_window - 1
            x_ = F.pad(x_, (pad_len, 0))
        x_ = self.conv1d(x_)
        return x_.permute(0, 2, 1)

    def _smfe(self, x, proj, se_block):
        x = self._temporal_fuse(x)
        x = proj(x)
        if se_block is not None:
            x = se_block(x)
        return x

    def _run_bi_encoder(self, base, f_enc, b_enc):
        out_f = f_enc(base)
        out_b = b_enc(torch.flip(base, [1]))
        out_b = torch.flip(out_b, [1])
        return torch.cat([out_f, out_b], dim=-1)

    # --------- forward ---------
    def forward(self, x):
        """
        x: (B, T, L, C) where L is either 9 or 4.
        If L==9, we internally slice levels [5,7,9,11]. If L==4, we assume that order.
        Returns:
          heads: (B, T, 4)  — per-stage logits
          h4:    (B, T, H or 2H) — final stage feature (for contrastive use)
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

        # Stage-wise SMFE
        s11 = self._smfe(b11, self.proj_11, self.se1)
        s09 = self._smfe(b9,  self.proj_09, self.se2)
        s07 = self._smfe(b7,  self.proj_07, self.se3)
        s05 = self._smfe(b5,  self.proj_05, self.se4)

        # Stage 1
        base1 = self.pos(s11)
        if self.use_bitrans:
            h1 = self._run_bi_encoder(base1, self.f_enc1, self.b_enc1)
        else:
            h1 = self.enc1(base1)
        head1 = self.head1(h1).squeeze(-1)

        # Stage 2
        h1_red = self.reduce_h(h1)
        base2 = self.fuse1(torch.cat([h1_red, s09], dim=-1))
        if self.use_bitrans:
            h2 = self._run_bi_encoder(base2, self.f_enc2, self.b_enc2)
        else:
            h2 = self.enc2(base2)
        head2 = self.head2(h2).squeeze(-1)

        # Stage 3
        h2_red = self.reduce_h(h2)
        base3 = self.fuse2(torch.cat([h2_red, s07], dim=-1))
        if self.use_bitrans:
            h3 = self._run_bi_encoder(base3, self.f_enc3, self.b_enc3)
        else:
            h3 = self.enc3(base3)
        head3 = self.head3(h3).squeeze(-1)

        # Stage 4
        h3_red = self.reduce_h(h3)
        base4 = self.fuse3(torch.cat([h3_red, s05], dim=-1))
        if self.use_bitrans:
            h4 = self._run_bi_encoder(base4, self.f_enc4, self.b_enc4)
        else:
            h4 = self.enc4(base4)
        head4 = self.head4(h4).squeeze(-1)

        heads = torch.stack([head1, head2, head3, head4], dim=-1)  # (B, T, 4)
        return heads, h4
