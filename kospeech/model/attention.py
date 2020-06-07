import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-product Attention

    Args:
        dim (int): dimention of attention

    Inputs: query, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, query, value):
        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    r"""
    Applies a multi-headed scaled dot mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )

    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value, prev_attn=None):
        batch_size = value.size(0)
        residual = query

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)

        context, attn = self.scaled_dot(query, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)
        combined = torch.cat([context, residual], dim=2)

        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1, self.hidden_dim)
        return output, attn


class LocationAwareAttention(nn.Module):
    """
    Applies a multi-headed location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    In the above paper applied a signle head, but we applied multi head concept.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution

    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """
    def __init__(self, hidden_dim, num_heads=8, conv_out_channel=10):
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.loc_projection = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.loc_conv = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.score_projection = nn.Linear(self.dim, 1, bias=True)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))

    def forward(self, query, value, prev_attn):
        batch_size, seq_len = value.size(0), value.size(1)
        residual = query

        # Initialize previous attn (alignment) to zeros
        if prev_attn is None:
            prev_attn = value.new_zeros(batch_size, self.num_heads, seq_len)

        # Calculate location energy
        loc_energy = torch.tanh(self.loc_projection(self.loc_conv(prev_attn).transpose(1, 2)))  # BxNxT => BxTxD
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)  # BxTxD => BxNxTxD

        # Projection & Shape matching
        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)  # Bx1xNxD
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)  # BxTxNxD
        query = query.contiguous().view(-1, 1, self.dim)        # BNx1xD
        value = value.contiguous().view(-1, seq_len, self.dim)  # BNxTxD

        # Get attention score, attn (align)
        score = self.score_projection(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)  # BNxT
        attn = F.softmax(score, dim=1)  # BNxT

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # BxTxNxD => BxNxTxD
        value = value.contiguous().view(-1, seq_len, self.dim)  # BxNxTxD => BNxTxD

        # Get context vector
        context = torch.bmm(attn.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)  # BNx1xT x BNxTxD => BxND
        attn = attn.view(batch_size, self.num_heads, -1)  # BNxT => BxNxT

        # Get output
        combined = torch.cat([context, residual], dim=2)
        output = self.out_projection(combined.view(-1, self.hidden_dim << 1)).view(batch_size, -1, self.hidden_dim)

        return output, attn


class CustomizingAttention(nn.Module):
    r"""
    Customizing Attention
    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    I combined these two attention mechanisms as custom.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The dimension of convolution
    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """

    def __init__(self, hidden_dim, num_heads=4, conv_out_channel=10):
        super(CustomizingAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=True)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.loc_conv = nn.Conv1d(in_channels=1, out_channels=conv_out_channel, kernel_size=3, padding=1)
        self.loc_projection = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value, prev_attn):
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)
        residual = query

        # Initialize previous attn (alignment) to zeros
        if prev_attn is None:
            prev_attn = value.new_zeros(batch_size, self.num_heads, v_len)

        loc_energy = self.get_loc_energy(prev_attn, batch_size, v_len)  # get location energy

        query = self.query_projection(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.value_projection(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, attn = self.scaled_dot(query, value)

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1, self.hidden_dim)

        return output, attn.squeeze()

    def get_loc_energy(self, prev_attn, batch_size, v_len):
        conv_feat = self.loc_conv(prev_attn.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.loc_projection(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy
