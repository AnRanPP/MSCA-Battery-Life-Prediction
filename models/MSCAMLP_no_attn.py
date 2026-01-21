# 保存为 F:\LXP\Project\PythonProject\BatteryLife\models\MSCAMLP_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MSCA_Plugin_no_attn import MSCAPlugin


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        out = self.in_linear(x)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        out = self.ln(self.dropout(out) + x)
        return out


class Model(nn.Module):
    """MSCA-MLP模型 - 使用MSCA插件替代CyclePatch"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.d_layers = configs.d_layers

        # MSCA Plugin for intra-cycle encoding
        self.msca_plugin = MSCAPlugin(configs)

        # Inter-cycle MLP encoder
        self.inter_flatten = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.early_cycle_threshold * self.d_model, self.d_model)
        )

        self.inter_MLP = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, self.drop_rate)
            for _ in range(self.d_layers)
        ])

        # Output head
        self.head_output = nn.Linear(self.d_model, 1)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False):
        """
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        """
        # Intra-cycle encoding with MSCA
        cycle_embeddings = self.msca_plugin(cycle_curve_data, curve_attn_mask)
        # cycle_embeddings: [B, early_cycle, d_model]

        # Inter-cycle processing with MLP
        inter_features = self.inter_flatten(cycle_embeddings)  # [B, d_model]

        for i in range(self.d_layers):
            inter_features = self.inter_MLP[i](inter_features)  # [B, d_model]

        # Final prediction
        preds = self.head_output(F.relu(inter_features))

        if return_embedding:
            return preds, inter_features
        else:
            return preds