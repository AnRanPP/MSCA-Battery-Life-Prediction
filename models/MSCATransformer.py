# 保存为 F:\LXP\Project\PythonProject\BatteryLife\models\MSCATransformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MSCA_Plugin import MSCAPlugin
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding


class Model(nn.Module):
    """MSCA-Transformer模型 - 使用MSCA插件替代CyclePatch"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout

        # MSCA Plugin for intra-cycle encoding
        self.msca_plugin = MSCAPlugin(configs)

        # Position encoding for transformer
        self.pe = PositionalEmbedding(self.d_model)

        # Inter-cycle Transformer encoder
        self.inter_TransformerEncoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Output layers
        self.inter_flatten = nn.Flatten(start_dim=1)
        self.head_output = nn.Linear(self.early_cycle_threshold * self.d_model, 1)

    def forward(self, cycle_curve_data, curve_attn_mask, return_embedding=False):
        """
        cycle_curve_data: [B, early_cycle, fixed_len, num_var]
        curve_attn_mask: [B, early_cycle]
        """
        # Intra-cycle encoding with MSCA
        cycle_embeddings = self.msca_plugin(cycle_curve_data, curve_attn_mask)
        # cycle_embeddings: [B, early_cycle, d_model]

        # Add position encoding
        cycle_embeddings = cycle_embeddings + self.pe(cycle_embeddings)

        # Prepare attention mask exactly like CPTransformer
        curve_attn_mask = curve_attn_mask.unsqueeze(1)  # [B, 1, L]
        curve_attn_mask = torch.repeat_interleave(curve_attn_mask, curve_attn_mask.shape[-1], dim=1)  # [B, L, L]
        curve_attn_mask = curve_attn_mask.unsqueeze(1)  # [B, 1, L, L]
        curve_attn_mask = curve_attn_mask == 0  # set True to mask

        # Inter-cycle processing with Transformer
        output, attns = self.inter_TransformerEncoder(cycle_embeddings, attn_mask=curve_attn_mask)

        # Flatten and predict
        output = self.inter_flatten(output)  # [B, early_cycle * d_model]
        preds = self.head_output(output)

        if return_embedding:
            return preds, output
        else:
            return preds