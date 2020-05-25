import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-product Attention

    Args:
        dim (int): dimention of attention

    Inputs: query, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, align
        - **context**: tensor containing the context vector from attention mechanism.
        - **align**: tensor containing the alignment from the encoder outputs.
    """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, value):
        attn_score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        align = F.softmax(attn_score, dim=2)
        context = torch.bmm(align, value)

        return context, align


class MultiHybridAttention(nn.Module):
    r"""
    Multi-Head + Location-Aware (Hybrid) Attention

    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    We combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        k (int): The dimension of convolution

    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output, aligng
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **align** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """

    def __init__(self, in_features, num_heads=8, k=10):
        super(MultiHybridAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.q_proj = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.v_proj = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.u_proj = nn.Linear(k, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))
        self.fc = nn.Linear(in_features << 1, in_features, bias=True)
        self.out_proj = nn.LayerNorm(in_features)

    def forward(self, query, value, prev_align):
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)
        residual = query

        loc_energy = self.get_loc_energy(prev_align, batch_size, v_len)  # get location energy

        query = self.q_proj(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.v_proj(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, align = self.scaled_dot(query, value)

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = self.out_proj(combined.view(-1, self.in_features << 1))
        output = self.normalize(output).view(batch_size, -1, self.in_features)

        return output, align.squeeze()

    def get_loc_energy(self, prev_align, batch_size, v_len):
        conv_feat = self.conv1d(prev_align.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.u_proj(conv_feat).view(batch_size, self.num_heads, v_len, self.dim).permute(0, 2, 1, 3)
        loc_energy = loc_energy.contiguous().view(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy
