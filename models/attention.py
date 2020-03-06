"""
Copyright 2020- Kai.Lib

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
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
    Applies an self attention mechanism on the output features from the decoder.
    output = tanh(W{concat(attn * encoder_output, decoder_output)} + b)

    Parameters
    -----------
        - decoder_hidden_size(int): The number of expected features in the output

    Inputs
    --------
        - **decoder_output** (batch, output_len, hidden_size): tensor containing the output features from the decoder.
        - **encoder_output** (batch, input_len, hidden_size): tensor containing features of the encoded input sequence.Steps to be maintained at a certain number to avoid extremely slow learning

    Returns
    ---------
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Reference
    ----------
        「A Structured Self-Attentive Sentence Embedding」
         https://arxiv.org/abs/1703.03130
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