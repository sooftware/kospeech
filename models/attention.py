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

class LocationAwareAttention(nn.Module):
    '''
    Applies an location-aware attention mechanism on the output features from the decoder.

    implementation of: https://arxiv.org/pdf/1506.07503.pdf
    '''
    def __init__(self, decoder_hidden_size, encoder_hidden_size, attn_size, conv_size=32, smoothing=False):
        super(LocationAwareAttention, self).__init__()
        self.attn_size = attn_size
        self.decoder_hidden_size = decoder_hidden_size
        self.conv_size = conv_size
        self.loc_conv = nn.Conv1d(in_channels=1, out_channels=conv_size, kernel_size=3, padding=1)
        self.attn_size = attn_size
        self.W = nn.Linear(decoder_hidden_size, attn_size, bias=False)
        self.V = nn.Linear(encoder_hidden_size, attn_size, bias=False)
        self.U = nn.Linear(conv_size, attn_size, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(attn_size).uniform_(-0.1, 0.1))
        self.w = nn.Linear(attn_size, 1, bias=False)
        self.smoothing = smoothing
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs, last_alignment):
        """ """
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)
        conv_feat = torch.transpose(self.loc_conv(last_alignment.unsqueeze(1)), 1, 2)

        attn_scores = self.w(self.tanh( self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.attn_size)
                                            + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.attn_size)
                                            + self.U(conv_feat)
                                            + self.b )).squeeze(dim=-1)

        if self.smoothing:
            attn_scores = torch.sigmoid(attn_scores)
            alignment = torch.div(attn_scores, attn_scores.sum(dim=-1).unsqueeze(dim=-1))
        else:
            alignment = self.softmax(attn_scores)

        output = torch.bmm(alignment.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)
        return output, alignment

class ConcatAttention(nn.Module):
    """
    Applies an concat attention mechanism on the output features from the decoder.

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
        super(ConcatAttention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)
        input_size = encoder_outputs.size(1)

        # get attention score
        attn_score = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        # get attention distribution
        attn_distribution = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # get attention value
        attn_val = torch.bmm(attn_distribution, encoder_outputs) # get attention value
        # concatenate attn_val & decoder_output
        combined = torch.cat((attn_val, decoder_output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output


class ContentBasedAttention(nn.Module):
    """ Applies an content-based attention mechanism on the output features from the decoder. """
    def __init__(self, decoder_hidden_size, encoder_hidden_size, attn_size):
        super(ContentBasedAttention, self).__init__()
        self.W = nn.Linear(decoder_hidden_size, attn_size, bias=False)
        self.V = nn.Linear(encoder_hidden_size, attn_size, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(attn_size).uniform_(-0.1, 0.1))
        self.w = nn.Linear(attn_size, 1, bias=False)
        self.attn_size = attn_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        attn_scores = self.w(self.tanh( self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.attn_size)
                                        + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.attn_size)
                                        + self.b )).squeeze(dim=-1)
        alignment = self.softmax(attn_scores)

        output = torch.bmm(alignment.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)
        return output