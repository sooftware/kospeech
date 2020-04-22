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
        >>> output = attention(decoder_output, encoder_outputs)
    """

    def __init__(self, in_features, n_head=4, dim=128):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.dim = dim
        self.W = nn.Linear(in_features, dim * n_head)
        self.V = nn.Linear(in_features, dim * n_head)
        self.fc = nn.Linear(in_features << 1, in_features)

    def forward(self, queries, values):
        batch_size = values.size(0)
        dec_length = queries.size(1)
        enc_length = values.size(1)

        preserved = queries

        queries = self.W(queries).view(batch_size, dec_length, self.n_head, self.dim)
        values = self.V(values).view(batch_size, enc_length, self.n_head, self.dim)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(-1, dec_length, self.dim)
        values = values.permute(2, 0, 1, 3).contiguous().view(-1, enc_length, self.dim)

        attn_score = torch.bmm(queries, values.transpose(1, 2))
        attn_distribution = F.softmax(attn_score, dim=2)

        context = torch.bmm(attn_distribution, values).view(self.n_head, batch_size, dec_length, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, dec_length, -1)

        combined = torch.cat([context, preserved], dim=2)
        output = torch.tanh(self.fc(combined.view(-1, 2 * self.in_features))).view(batch_size, -1, self.in_features)

        return output
