import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Optional, Tuple
from kospeech.models.seq2seq.modules import Linear, LayerNorm


class AddNorm(nn.Module):
    """
    Add & Normalization layer proposed in "Attention Is All You Need".
    Transformer employ a residual connection around each of the two sub-layers,
    (Multi-Head Attention & Feed-Forward) followed by layer normalization.
    """
    def __init__(self, sublayer: nn.Module, d_model: int = 512) -> None:
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)

        if isinstance(output, tuple):
            return self.layer_norm(output[0] + residual), output[1]
        else:
            return self.layer_norm(output + residual)


# class ScaledDotProductAttention(nn.Module):
#     """
#     Scaled Dot-Product Attention proposed in "Attention Is All You Need"
#     Compute the dot products of the query with all keys, divide each by sqrt(dim),
#     and apply a softmax function to obtain the weights on the values
#
#     Args: dim, mask
#         dim (int): dimention of attention
#         mask (torch.Tensor): tensor containing indices to be masked
#
#     Inputs: query, key, value, mask
#         - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
#         - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
#         - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
#         - **mask** (-): tensor containing indices to be masked
#
#     Returns: context, attn
#         - **context**: tensor containing the context vector from attention mechanism.
#         - **attn**: tensor containing the attention (alignment) from the encoder outputs.
#     """
#     def __init__(self, dim: int) -> None:
#         super(ScaledDotProductAttention, self).__init__()
#         self.sqrt_dim = np.sqrt(dim)
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
#         score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
#
#         if mask is not None:
#             score.masked_fill_(mask.view(score.size()), -1e9)
#
#         attn = F.softmax(score, -1)
#         context = torch.bmm(attn, value)
#         return context, attn
#
#
# class MultiHeadAttention(nn.Module):
#     """
#     Multi-Head Attention proposed in "Attention Is All You Need"
#     Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
#     project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
#     These are concatenated and once again projected, resulting in the final values.
#     Multi-head attention allows the model to jointly attend to information from different representation
#     subspaces at different positions.
#
#     MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
#         where head_i = Attention(Q · W_q, K · W_k, V · W_v)
#
#     Args:
#         d_model (int): The dimension of keys / values / quries (default: 512)
#         num_heads (int): The number of attention heads. (default: 8)
#
#     Inputs: query, key, value, mask
#         - **query** (batch, q_len, d_model): In transformer, three different ways:
#             Case 1: come from previoys decoder layer
#             Case 2: come from the input embedding
#             Case 3: come from the output embedding (masked)
#
#         - **key** (batch, k_len, d_model): In transformer, three different ways:
#             Case 1: come from the output of the encoder
#             Case 2: come from the input embeddings
#             Case 3: come from the output embedding (masked)
#
#         - **value** (batch, v_len, d_model): In transformer, three different ways:
#             Case 1: come from the output of the encoder
#             Case 2: come from the input embeddings
#             Case 3: come from the output embedding (masked)
#
#         - **mask** (-): tensor containing indices to be masked
#
#     Returns: output, attn
#         - **output** (batch, output_len, dimensions): tensor containing the attended output features.
#         - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
#     """
#     def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
#         super(MultiHeadAttention, self).__init__()
#
#         assert d_model % num_heads == 0, "d_model % num_heads should be zero."
#
#         self.d_head = int(d_model / num_heads)
#         self.num_heads = num_heads
#         self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
#         self.linear_q = Linear(d_model, self.d_head * num_heads)
#         self.linear_k = Linear(d_model, self.d_head * num_heads)
#         self.linear_v = Linear(d_model, self.d_head * num_heads)
#         self.linear = Linear(d_model, d_model)
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
#         batch_size = value.size(0)
#
#         query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
#         key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
#         value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD
#
#         query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
#         key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
#         value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD
#
#         if mask is not None:
#             mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN
#
#         context, attn = self.scaled_dot_attn(query, key, value, mask)
#         context = context.view(self.num_heads, batch_size, -1, self.d_head)
#         context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND
#
#         output = self.linear(context)
#         return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    """
    Position-wise Feedforward Networks proposed in "Attention Is All You Need".
    Fully connected feed-forward network, which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    Another way of describing this is as two convolutions with kernel size 1.
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048,
                 dropout_p: float = 0.3, ffnet_style: str = 'feed_forward') -> None:
        super(PoswiseFeedForwardNet, self).__init__()
        self.ffnet_style = ffnet_style.lower()
        if self.ffnet_style == 'ff':
            self.feed_forward = nn.Sequential(
                Linear(d_model, d_ff),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                Linear(d_ff, d_model),
                nn.Dropout(dropout_p)
            )

        elif self.ffnet_style == 'conv':
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        else:
            raise ValueError("Unsupported mode: {0}".format(self.mode))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.ffnet_style == 'ff':
            output = self.feed_forward(inputs)

        else:
            output = self.conv1(inputs.transpose(1, 2))
            output = self.relu(output)
            output = self.conv2(output).transpose(1, 2)

        return output
