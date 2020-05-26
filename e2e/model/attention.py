import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    def __init__(self, in_features, num_heads=8, num_kernels=10):
        super().__init__()
        self.num_heads = num_heads
        self.dim = int(in_features / num_heads)
        self.conv1d = nn.Conv1d(num_heads, num_kernels)
        self.linear_u = nn.Linear(num_kernels, self.dim, bias=False)
        self.linear_q = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.linear_v = nn.Linear(in_features, self.dim * num_heads, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))
        self.linear_out = nn.Linear(in_features << 1, in_features, bias=True)

    def forward(self, query, key, value, prev_align):  # value : BxTxD
        batch_size, k_len, q_len = key.size(0), key.size(1), query.size(1)

        if prev_align is None:
            prev_align = key.new_zeros(batch_size, self.num_heads, k_len)  # BxNxT

        loc_energy = torch.tanh(self.linear_u(self.conv1d(prev_align).transpose(1, 2)))  # BxNxT => BxTxD
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, k_len, self.dim)  # BxNxTxD => BNxTxD

        query = self.linear_q(query).view(batch_size, q_len, self.num_heads, self.dim)      # BxTxNxD
        query = query.permute(0, 2, 1, 3).view(-1, q_len, self.dim)  # BxNxTxD => BNx1xD

        key = self.linear_v(key).view(batch_size, k_len, self.num_heads, self.dim)      # BxTxNxD
        key = key.permute(0, 2, 1, 3).view(-1, k_len, self.dim)  # BxNxTxD =>BNxTxD

        value = value.view(batch_size, k_len, self.num_heads, self.dim)
        value = value.permute(0, 2, 1, 3).view(-1, k_len, self.dim)  # BxNxTxD => BNxTxD

        score = self.linear_out(torch.tanh(key + query + loc_energy + self.bias)).squeeze(2)  # BNxTxD => BNxT
        align = F.softmax(score, dim=2)  # BNxT

        context = torch.bmm(align.unsqueeze(1), value).squeeze(1)  # BNx1xT x BNxTxD => BNxD
        align = align.view(batch_size, self.num_heads, -1)         # BNxT => BxNxT

        return context, align
