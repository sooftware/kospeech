import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    r"""Applies a multi-head attention mechanism on the output features from the decoder.

    Refer to 「State-of-the-art Speech Recognition With Sequence-to-Sequence Models」 Paper
    https://arxiv.org/abs/1712.01769


    Args:
        in_features (int): The number of expected features in the output
        num_head (int): number of heads. (default: 4)
        dim (int): dimension size of sub heads. (default: 128)

    Inputs: query, key
        - **query** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **key** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Examples::
        >>> attention = MultiHeadAttention(in_features=512, n_head=8, dim=64)
        >>> output = attention(Q, K, V)
    """

    def __init__(self, in_features, num_head=8, dim=64):
        super(MultiHeadAttention, self).__init__()

        assert num_head * dim == in_features, "<num_head> * <dim> size must be same to <in_features> size"

        self.in_features = in_features
        self.num_head = num_head
        self.dim = dim
        self.W_Q = nn.Linear(in_features, dim * num_head)
        self.W_K = nn.Linear(in_features, dim * num_head)
        self.W_V = nn.Linear(in_features, dim * num_head)
        self.fc = nn.Linear(in_features << 1, in_features)

    def forward(self, Q, K, V):
        batch_size = V.size(0)
        q_len = Q.size(1)
        k_len = K.size(1)

        residual = Q

        q_s = self.W_Q(Q).view(batch_size, q_len, self.num_head, self.dim).permute(2, 0, 1, 3)
        k_s = self.W_K(K).view(batch_size, k_len, self.num_head, self.dim).permute(2, 0, 1, 3)
        v_s = self.W_V(V).view(batch_size, k_len, self.num_head, self.dim).permute(2, 0, 1, 3)

        q_s = q_s.contiguous().view(-1, q_len, self.dim)
        k_s = k_s.contiguous().view(-1, k_len, self.dim)
        v_s = v_s.contiguous().view(-1, k_len, self.dim)

        score = torch.bmm(q_s, k_s.transpose(1, 2)) / np.sqrt(self.dim)  # scaled dot-product
        align = F.softmax(score, dim=2)

        context = torch.bmm(align, v_s).view(self.num_head, batch_size, q_len, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = torch.tanh(self.fc(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)

        return output
