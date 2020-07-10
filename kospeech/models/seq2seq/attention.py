import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from kospeech.models.seq2seq.modules import Linear, LayerNorm


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, value
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing projection vector for encoder.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Applies a multi-headed scaled dot mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.

    Args:
        d_model (int): dimension of model
        num_heads (int): The number of heads. (default: )

    Inputs: query, value
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context
        - **context** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769

    Contributor:
        - Soohwan Kim @sooftware
        - Deokjin Seo @qute012
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.linear_q = Linear(d_model, self.d_head * num_heads)
        self.linear_k = Linear(d_model, self.d_head * num_heads)
        self.linear_v = Linear(d_model, self.d_head * num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        context, attn = self.scaled_dot_attn(query, key, value)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND
        return context, attn


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        hidden_dim (int): The number of expected features in the output
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, hidden_dim: int = 512, smoothing: bool = True) -> None:
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.linear_q = Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.linear(torch.tanh(
                self.linear_q(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.linear_v(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))

        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD
        context = torch.cat([context, query.squeeze(1)], dim=1)

        return context, attn
