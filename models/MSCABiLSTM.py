# 保存为 F:\LXP\Project\PythonProject\BatteryLife\models\MSCABiLSTM.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MSCA_Plugin import MSCAPlugin


class Model(nn.Module):
    """MSCA-BiLSTM模型 - 使用MSCA插件替代CyclePatch"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        self.d_layers = configs.d_layers

        # MSCA Plugin for intra-cycle encoding
        self.msca_plugin = MSCAPlugin(configs)

        # Inter-cycle BiLSTM encoder
        self.inter_BiLSTM = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_ff // 2,  # Divided by 2 for bidirectional
            num_layers=self.d_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.drop_rate if self.d_layers > 1 else 0
        )

        # Output head
        self.head_output = nn.Linear(self.d_ff, 1)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False):
        """
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        """
        # Intra-cycle encoding with MSCA
        cycle_embeddings = self.msca_plugin(cycle_curve_data, curve_attn_mask)
        # cycle_embeddings: [B, early_cycle, d_model]

        # Get sequence lengths for packing
        lengths = curve_attn_mask.sum(dim=1).cpu()

        # Pack sequences for BiLSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            cycle_embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # Inter-cycle processing with BiLSTM
        packed_output, (h_n, c_n) = self.inter_BiLSTM(packed_input)

        # Concatenate forward and backward hidden states
        h_forward = h_n[-2]  # Last layer, forward
        h_backward = h_n[-1]  # Last layer, backward
        inter_features = torch.cat([h_forward, h_backward], dim=-1)  # [B, d_ff]

        # Final prediction
        preds = self.head_output(inter_features)

        if return_embedding:
            return preds, inter_features
        else:
            return preds