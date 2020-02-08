"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder. (Concat-Attention)

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * encoder_output) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: decoder_output, encoder_output
        - **decoder_output** (batch, output_len, hidden_size): tensor containing the output features from the decoder.
        - **encoder_output** (batch, input_len, hidden_size): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, decoder_output, encoder_output):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)
        input_size = encoder_output.size(1)

        attn_score = torch.bmm(decoder_output, encoder_output.transpose(1, 2))
        attn_distribution = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        attn_val = torch.bmm(attn_distribution, encoder_output) # get attention value

        combined = torch.cat((attn_val, decoder_output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output
