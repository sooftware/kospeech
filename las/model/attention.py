import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    r"""
    Applies a multi-head attention mechanism on the output features from the decoder.
    This attention proposed in "Attention Is All You Need" paper.
    We refer to "State-of-the-art Speech Recognition With Sequence-to-Sequence Models" paper.
    This paper applied Multi-Head attention in speech recognition task.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): number of heads. (default: 4)

    Inputs: Q, V
        - **Q** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **V** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Examples::
        >>> attention = MultiHeadAttention(in_features=512, n_head=4)
        >>> output = attention(Q, V)
    """

    def __init__(self, in_features, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.W_Q = nn.Linear(in_features, self.dim * num_heads)
        self.W_V = nn.Linear(in_features, self.dim * num_heads)
        self.fc = nn.Linear(in_features << 1, in_features)

    def forward(self, Q, V):
        batch_size = V.size(0)
        q_len = Q.size(1)
        v_len = V.size(1)

        residual = Q

        q_s = self.W_Q(Q).view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        v_s = self.W_V(V).view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)

        q_s = q_s.contiguous().view(-1, q_len, self.dim)
        v_s = v_s.contiguous().view(-1, v_len, self.dim)

        score = torch.bmm(q_s, v_s.transpose(1, 2)) / np.sqrt(self.dim)  # scaled dot-product
        align = F.softmax(score, dim=2)

        context = torch.bmm(align, v_s).view(self.num_heads, batch_size, q_len, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = torch.tanh(self.fc(combined.view(-1, self.in_features << 1))).view(batch_size, -1, self.in_features)

        return output
