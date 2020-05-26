import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    r"""
    Multi-headed Location-Aware (Hybrid) Attention
    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    We combined these two attention mechanisms as custom.

    Args:
        in_features (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution

    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_align** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: context, align
        - **context** (batch, output_len, dimensions): tensor containing the attention context vector
        - **align** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """
    def __init__(self, in_features, num_heads=8, conv_out_channel=10):
        super().__init__()
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.conv1d = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.linear_q = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.linear_v = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.linear_u = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))
        self.scoring = nn.Linear(self.dim, 1, bias=True)

    def forward(self, query, value, prev_align):
        batch_size, seq_len = value.size(0), value.size(1)

        # Initialize previous alignment to zeros
        if prev_align is None:
            prev_align = value.new_zeros(batch_size, self.num_heads, seq_len)

        # Calculate location energy
        loc_energy = torch.tanh(self.linear_u(self.conv1d(prev_align).transpose(1, 2)))
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        query = query.contiguous().view(-1, 1, self.dim)
        value = value.contiguous().view(-1, seq_len, self.dim)

        score = self.scoring(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)
        align = F.softmax(score, dim=1)

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = value.contiguous().view(-1, seq_len, self.dim)

        context = torch.bmm(align.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)
        align = align.view(batch_size, self.num_heads, -1)

        return context, align
