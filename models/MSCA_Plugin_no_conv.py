# F:\LXP\Project\PythonProject\BatteryLife\models\MSCA_Plugin_no_conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 以下 MultiScaleConvBlock, CycleAttention, AdaptiveGating 类与原始文件完全相同 ---
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiScaleConvBlock, self).__init__()
        n_kernels = len(kernel_sizes)
        ch_per_kernel = out_channels // n_kernels
        remainder = out_channels % n_kernels
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_ch = ch_per_kernel + (remainder if i == 0 else 0)
            self.convs.append(nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=k // 2))
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        out = torch.cat(outputs, dim=1)
        return F.relu(self.bn(out))


class CycleAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(CycleAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B, D = x.shape
        Q = self.W_q(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, D)
        output = self.W_o(context)
        output = self.ln(output + x)
        return output


class AdaptiveGating(nn.Module):
    def __init__(self, d_model):
        super(AdaptiveGating, self).__init__()
        self.gate_net = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x1, x2):
        gate = self.gate_net(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2


# --- Helper 类结束 ---

class MSCABlock(nn.Module):
    """增强的MSCA块（此版本用于 'w/o Conv' 实验）"""

    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate, use_conv=True):
        super(MSCABlock, self).__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.conv_path = MultiScaleConvBlock(3, hidden_dim)
        self.linear_path = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.attention = CycleAttention(hidden_dim, n_heads=8, dropout=drop_rate)
        if self.use_conv:
            self.adaptive_gate = AdaptiveGating(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, raw_cycle_data=None):
        B, S, D = x.shape
        linear_features = self.linear_path(x)

        # 对于这个实验，我们永远只走线性路径
        fused_features = linear_features

        enhanced_features = []
        for i in range(S):
            cycle_feat = fused_features[:, i, :]
            cycle_feat = self.attention(cycle_feat)
            enhanced_features.append(cycle_feat)
        enhanced_features = torch.stack(enhanced_features, dim=1)

        out = self.out_proj(enhanced_features)
        residual = self.residual_proj(x)
        return out + residual


class MSCAPlugin(nn.Module):
    """MSCA插件 ('w/o Conv' 版本)"""

    def __init__(self, configs):
        super(MSCAPlugin, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers
        self.intra_flatten = nn.Flatten(start_dim=2)
        input_dim = self.charge_discharge_length * 3
        self.intra_embed = nn.Linear(input_dim, self.d_model)
        self.msca_layers = nn.ModuleList()
        for i in range(self.e_layers):
            # --- 核心修改说明 ---
            # 强制所有层都不使用卷积
            use_conv = False
            # --- 修改结束 ---
            self.msca_layers.append(
                MSCABlock(
                    self.d_model,
                    self.d_ff,
                    self.d_model,
                    self.drop_rate,
                    use_conv=use_conv
                )
            )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.early_cycle_threshold, self.d_model) * 0.02
        )
        self.cycle_norm = nn.LayerNorm(self.d_model)

    def forward(self, cycle_curve_data, curve_attn_mask):
        raw_cycle_data = cycle_curve_data.clone()
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask == 0] = 0
        raw_cycle_data[tmp_curve_attn_mask == 0] = 0
        cycle_embeddings = self.intra_flatten(cycle_curve_data)
        cycle_embeddings = self.intra_embed(cycle_embeddings)
        cycle_embeddings = cycle_embeddings + self.pos_encoding

        # --- 核心修改说明 ---
        # 由于所有层的 use_conv 都为 False，我们无需传入 raw_cycle_data
        for layer in self.msca_layers:
            cycle_embeddings = layer(cycle_embeddings, None)
        # --- 修改结束 ---

        cycle_embeddings = self.cycle_norm(cycle_embeddings)
        mask = curve_attn_mask.unsqueeze(-1)
        cycle_embeddings = cycle_embeddings * mask
        return cycle_embeddings