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
        n_head (int): number of heads. (default: 4)
        dim (int): dimension size of sub heads. (default: 128)

    Inputs: query, key
        - **query** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **key** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Examples::
        >>> attention = MultiHeadAttention(in_features, n_head=4, dim=128)
        >>> output = attention(query, key)
    """
    def __init__(self, in_features, n_head = 4, dim = 128):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.linear_q = nn.Linear(in_features, dim * n_head)
        self.linear_k = nn.Linear(in_features, dim * n_head)
        self.n_head = n_head
        self.dim = dim
        self.out = nn.Linear(in_features << 1, in_features)


    def forward(self, query, key):
        batch_size = key.size(0)
        query_length = query.size(1)
        key_length = key.size(1)

        preserved = query

        query = self.linear_q(query).view(batch_size, query_length, self.n_head, self.dim).permute(2, 0, 1, 3)
        key = self.linear_k(key).view(batch_size, key_length, self.n_head, self.dim).permute(2, 0, 1, 3)

        query = query.contiguous().view(-1, query_length, self.dim)  # -1 = n_head * batch_size
        key = key.contiguous().view(-1, key_length, self.dim)

        # get attention score
        attn_score = torch.bmm(query, key.transpose(1, 2))

        # get attention distribution
        attn_distribution = F.softmax(attn_score, dim=2)
        
        # get context vector
        context = torch.bmm(attn_distribution, key).view(self.n_head, batch_size, query_length, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, query_length, -1)
        
        # concatenate context & query
        combined = torch.cat([context, preserved], dim=2)
        output = torch.tanh(self.out(combined.view(-1, 2 * self.in_features))).view(batch_size, -1, self.in_features)

        return output