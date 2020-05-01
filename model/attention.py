import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    r"""
    Applies an multi-head attention mechanism on the output features from the decoder.

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
        >>> output = attention(queries, values)
    """

    def __init__(self, in_features, num_head=8, dim=64):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.num_head = num_head
        self.dim = dim
        self.W_Q = nn.Linear(in_features, dim * num_head)
        self.W_V = nn.Linear(in_features, dim * num_head)
        self.fc = nn.Linear(in_features + dim * num_head, in_features)

    def forward(self, Q, V):
        batch_size = V.size(0)
        residual = Q

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_head, self.dim)
        v_s = self.W_V(V).view(batch_size, -1, self.num_head, self.dim)

        q_s = q_s.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)
        v_s = v_s.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_head, -1, self.dim)

        attn_score = torch.bmm(q_s, v_s.transpose(1, 2))
        align = F.softmax(attn_score, dim=2)

        attn_val = torch.bmm(align, v_s).view(self.num_head, batch_size, -1, self.dim)
        attn_val = attn_val.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_head * self.dim)
        combined = torch.cat([attn_val, residual], dim=2)

        context = torch.tanh(self.fc(combined.view(-1, self.in_features + self.dim * self.num_head)))
        context = context.view(batch_size, -1, self.in_features)

        return context
