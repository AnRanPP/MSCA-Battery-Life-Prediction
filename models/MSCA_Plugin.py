# 调试版本的 F:\LXP\Project\PythonProject\BatteryLife\models\MSCA_Plugin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块，用于提取不同粒度的循环特征"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiScaleConvBlock, self).__init__()
        # 确保输出通道数能被kernel数量整除
        n_kernels = len(kernel_sizes)
        ch_per_kernel = out_channels // n_kernels
        remainder = out_channels % n_kernels

        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # 给第一个卷积分配额外的通道（如果有余数）
            out_ch = ch_per_kernel + (remainder if i == 0 else 0)
            self.convs.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=k // 2)
            )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # DEBUG: 打印输入形状
        # print(f"MultiScaleConvBlock input shape: {x.shape}")
        # x should be: [B*cycles, channels, length]
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        out = torch.cat(outputs, dim=1)
        return F.relu(self.bn(out))


class CycleAttention(nn.Module):
    """循环内注意力机制"""

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

        # Self-attention
        Q = self.W_q(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, D)
        output = self.W_o(context)

        # Residual connection
        output = self.ln(output + x)
        return output


class AdaptiveGating(nn.Module):
    """自适应门控机制"""

    def __init__(self, d_model):
        super(AdaptiveGating, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gate = self.gate_net(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2


class MSCABlock(nn.Module):
    """增强的MSCA块，结合多尺度、注意力和自适应机制"""

    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate, use_conv=True):
        super(MSCABlock, self).__init__()

        self.use_conv = use_conv

        # Multi-scale convolution path (if enabled)
        if self.use_conv:
            self.conv_path = MultiScaleConvBlock(3, hidden_dim)  # 3 channels for V, Q, dQ/dV

        # Linear transformation path
        self.linear_path = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Attention enhancement
        self.attention = CycleAttention(hidden_dim, n_heads=8, dropout=drop_rate)

        # Adaptive fusion (only if conv is used)
        if self.use_conv:
            self.adaptive_gate = AdaptiveGating(hidden_dim)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, raw_cycle_data=None):
        """
        x: [B, cycles, flattened_dim] - flattened cycle data
        raw_cycle_data: [B, cycles, length, channels] - original cycle data for conv processing
        """
        B, S, D = x.shape

        # Linear transformation path
        linear_features = self.linear_path(x)  # [B, S, hidden_dim]

        # Multi-scale convolution path (if raw data provided and conv enabled)
        if self.use_conv and raw_cycle_data is not None:
            # DEBUG: 打印原始数据形状
            # print(f"raw_cycle_data shape: {raw_cycle_data.shape}")

            # Reshape for convolution
            _, _, L, C = raw_cycle_data.shape
            # Conv1d expects [batch, channels, length]
            # We have [B, S, L, C] -> need [B*S, C, L]

            # 使用permute确保维度转换正确
            conv_input = raw_cycle_data.reshape(B * S, L, C).permute(0, 2, 1).contiguous()  # [B*S, C, L]
            # 添加调试打印 - 注意这里expected应该是 C, L 而不是 L, C
            print(f"DEBUG MSCA: conv_input shape after permute: {conv_input.shape}, expected: [{B*S}, {C}, {L}]")

            # 额外验证
            if conv_input.shape != (B * S, C, L):
                print(f"ERROR: Permute failed! Shape is {conv_input.shape}, but should be ({B * S}, {C}, {L})")
                # 尝试另一种方法
                conv_input = raw_cycle_data.reshape(B * S, L, C)
                conv_input = conv_input.transpose(1, 2).contiguous()
                print(f"After transpose fix: {conv_input.shape}")

            # 确保维度正确
            if conv_input.shape != (B * S, C, L):
                print(f"ERROR: Conv input shape is {conv_input.shape}, expected {(B * S, C, L)}")
                print(f"B={B}, S={S}, C={C}, L={L}")

            conv_features = self.conv_path(conv_input)  # [B*S, hidden_dim, L]

            # Global average pooling
            conv_features = F.adaptive_avg_pool1d(conv_features, 1).squeeze(-1)  # [B*S, hidden_dim]
            conv_features = conv_features.view(B, S, -1)  # [B, S, hidden_dim]

            # Adaptive fusion of linear and conv features
            fused_features = self.adaptive_gate(linear_features, conv_features)
        else:
            fused_features = linear_features

        # Apply attention to each cycle
        enhanced_features = []
        for i in range(S):
            cycle_feat = fused_features[:, i, :]  # [B, hidden_dim]
            cycle_feat = self.attention(cycle_feat)  # [B, hidden_dim]
            enhanced_features.append(cycle_feat)

        enhanced_features = torch.stack(enhanced_features, dim=1)  # [B, S, hidden_dim]

        # Output projection with residual
        out = self.out_proj(enhanced_features)
        residual = self.residual_proj(x)

        return out + residual


class MSCAPlugin(nn.Module):
    """MSCA插件 - 可替代CyclePatch的intra-cycle encoder"""

    def __init__(self, configs):
        super(MSCAPlugin, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.e_layers = configs.e_layers

        # Initial embedding
        self.intra_flatten = nn.Flatten(start_dim=2)
        input_dim = self.charge_discharge_length * 3
        self.intra_embed = nn.Linear(input_dim, self.d_model)

        # MSCA layers - 第一层使用conv，后续层不使用
        self.msca_layers = nn.ModuleList()
        for i in range(self.e_layers):
            use_conv = (i == 0)  # 只在第一层使用卷积
            self.msca_layers.append(
                MSCABlock(
                    self.d_model,
                    self.d_ff,
                    self.d_model,
                    self.drop_rate,
                    use_conv=use_conv
                )
            )

        # Learnable position encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.early_cycle_threshold, self.d_model) * 0.02
        )

        # Cycle-level normalization
        self.cycle_norm = nn.LayerNorm(self.d_model)

    def forward(self, cycle_curve_data, curve_attn_mask):
        """
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        返回: [B, early_cycle, d_model] - 处理后的循环嵌入
        """
        # Store original data for multi-scale processing
        raw_cycle_data = cycle_curve_data.clone()

        # Mask unseen data
        tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        cycle_curve_data[tmp_curve_attn_mask == 0] = 0
        raw_cycle_data[tmp_curve_attn_mask == 0] = 0

        # Flatten and embed
        cycle_embeddings = self.intra_flatten(cycle_curve_data)  # [B, early_cycle, fixed_len * num_var]
        cycle_embeddings = self.intra_embed(cycle_embeddings)  # [B, early_cycle, d_model]

        # Add position encoding
        cycle_embeddings = cycle_embeddings + self.pos_encoding

        # Apply MSCA layers
        for i, layer in enumerate(self.msca_layers):
            # Pass raw data only to first layer (which has conv enabled)
            if i == 0:
                cycle_embeddings = layer(cycle_embeddings, raw_cycle_data)
            else:
                cycle_embeddings = layer(cycle_embeddings, None)

        # Final normalization
        cycle_embeddings = self.cycle_norm(cycle_embeddings)

        # Mask invalid cycles
        mask = curve_attn_mask.unsqueeze(-1)
        cycle_embeddings = cycle_embeddings * mask

        return cycle_embeddings