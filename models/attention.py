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
    Applies an attention mechanism on the output features from the decoder.

    Args:
        score_function (str) : type of Attention`s score function (default 'hybrid')
            list => 'hybrid', 'content-based', 'dop-product'
    """
    def __init__(self, score_function = 'hybrid', decoder_hidden_size = None):
        super(Attention, self).__init__()
        if score_function.lower() == 'hybrid':
            self.attention = HybridAttention(
                decoder_hidden_size = decoder_hidden_size,
                encoder_hidden_size = decoder_hidden_size,
                #context_size = int(decoder_hidden_size >> 1),
                context_size=2040,
                k = 10,
                smoothing = True
            )
        elif score_function.lower() == 'content-based':
            self.attention = ContentBasedAttention(
                decoder_hidden_size = decoder_hidden_size,
                encoder_hidden_size = decoder_hidden_size,
                context_size = decoder_hidden_size
            )
        elif score_function.lower() == 'dot-product':
            self.attention = DotProductAttention(decoder_hidden_size = decoder_hidden_size)
        else:
            raise ValueError("Invalid Score function !!")

    def forward(self, decoder_output, encoder_outputs, last_alignment):
        if isinstance(self.attention, HybridAttention):
            context, alignment = self.attention.forward(decoder_output, encoder_outputs, last_alignment)
            return context, alignment
        else:
            context = self.attention.forward(decoder_output, encoder_outputs)
            return context


class HybridAttention(Attention):
    '''
    Score function : Hybrid attention (Location-aware Attention)

    .. math ::
        score = w^T( tanh( Ws + Vhs + Uf + b ) )
            => s : decoder_output
               hs : encoder_outputs
               f : loc_conv(last_alignment)
               b : bias

    Reference:
        「Attention-Based Models for Speech Recognition」 Paper
         https://arxiv.org/pdf/1506.07503.pdf
    '''
    def __init__(self, decoder_hidden_size, encoder_hidden_size, context_size, k = 10, smoothing=True):
        super(Attention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.context_size = context_size
        self.loc_conv = nn.Conv1d(in_channels=1, out_channels=k, kernel_size=3, padding=1)
        self.W = nn.Linear(decoder_hidden_size, context_size, bias=False)
        self.V = nn.Linear(encoder_hidden_size, context_size, bias=False)
        self.U = nn.Linear(k, context_size, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(context_size).uniform_(-0.1, 0.1))
        self.w = nn.Linear(context_size, 1, bias=False)
        self.smoothing = smoothing
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs, last_align):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        if last_align is None:
            attn_scores = self.w(
                self.tanh(
                    self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.b
                )
            ).squeeze(dim=-1)
        else:
            last_align = torch.transpose(self.loc_conv(last_align.unsqueeze(1)), 1, 2)
            attn_scores = self.w(
                self.tanh(
                    self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                    + self.U(last_align)
                    + self.b
                )
            ).squeeze(dim=-1)

        if self.smoothing:
            attn_scores = torch.sigmoid(attn_scores)
            alignment = torch.div(attn_scores, attn_scores.sum(dim=-1).unsqueeze(dim=-1))
        else:
            alignment = self.softmax(attn_scores)

        context = torch.bmm(alignment.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)
        return context, alignment


class ContentBasedAttention(Attention):
    """ Applies an content-based attention mechanism on the output features from the decoder. """
    def __init__(self, decoder_hidden_size, encoder_hidden_size, context_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(decoder_hidden_size, context_size, bias=False)
        self.V = nn.Linear(encoder_hidden_size, context_size, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(context_size).uniform_(-0.1, 0.1))
        self.w = nn.Linear(context_size, 1, bias=False)
        self.context_size = context_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        hidden_size = decoder_output.size(2)

        attn_scores = self.w(
            self.tanh(
                self.W(decoder_output.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                + self.V(encoder_outputs.reshape(-1, hidden_size)).view(batch_size, -1, self.context_size)
                + self.b
            )
        ).squeeze(dim=-1)
        alignment = self.softmax(attn_scores)
        context = torch.bmm(alignment.unsqueeze(dim=1), encoder_outputs).squeeze(dim=1)

        return context


class DotProductAttention(Attention):
    """
    Applies an dot product attention mechanism on the output features from the decoder.

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
        - **encoder_output** (batch, input_len, hidden_size): tensor containing features of the encoded input sequence.Steps to be maintained at a certain number to avoid extremely slow learning

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """
    def __init__(self, decoder_hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(decoder_hidden_size*2, decoder_hidden_size)

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
        context = torch.tanh(self.W(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return context