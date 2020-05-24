import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-product Attention

    Args:
        dim (int): dimention of attention

    Inputs: Q, V
        - **Q** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **V** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, align
        - **context**: context vector
        - **align**: alignment
    """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value):
        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        align = self.softmax(score)
        context = torch.bmm(align, value)
        return context, align


class MultiLocAwareAttention(nn.Module):
    r"""
    Multi-Head + Location-Aware Attention

    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    We combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        k (int): The dimension of convolution

    Inputs: query, value, prev_align
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """

    def __init__(self, in_features, num_heads=8, k=10):
        super(MultiLocAwareAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W_Q = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.W_V = nn.Linear(in_features, self.dim * num_heads, bias=True)
        self.W_U = nn.Linear(k, self.dim, bias=True)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.fc = nn.Linear(in_features << 1, in_features, bias=True)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, query, value, prev_align):  # (batch_size * num_heads, v_len)
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)
        residual = query

        loc_energy = self.get_loc_energy(prev_align, batch_size, v_len)

        q_s = self.W_Q(query).view(batch_size, q_len, self.num_heads * self.dim)
        v_s = self.W_V(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy

        q_s = q_s.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3).reshape(-1, q_len, self.dim)
        v_s = v_s.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3).reshape(-1, v_len, self.dim)

        context, align = self.scaled_dot(q_s, v_s)

        context = context.view(self.num_heads, batch_size, q_len, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = self.norm(self.fc(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)

        return output, align.squeeze()

    def get_loc_energy(self, prev_align, batch_size, v_len):
        conv_feat = self.conv1d(prev_align.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.W_U(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy
