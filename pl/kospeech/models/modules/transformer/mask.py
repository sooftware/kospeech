# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor
from typing import Any, Optional


def get_non_pad_mask(inputs: Tensor, input_lengths: Tensor) -> Tensor:
    """ Padding position is set to 0, either use input_lengths or pad_id """
    batch_size = inputs.size(0)

    if len(inputs.size()) == 2:
        non_pad_mask = inputs.new_ones(inputs.size())  # B x T
    else:
        non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # B x T

    for i in range(batch_size):
        non_pad_mask[i, input_lengths[i]:] = 0

    return non_pad_mask.unsqueeze(-1)


def get_decoder_self_attn_mask(seq_k: Tensor, seq_q: Tensor, pad_id):
    """ For masking the decoder self attention """
    def _get_attn_key_pad_mask(seq_k, seq_q, pad_id):
        """ For masking out the padding part of key sequence. """
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(pad_id)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask

    def _get_subsequent_mask(inputs: Tensor) -> Tensor:
        """ Makes subsequent masking """
        batch_size, seq_length = inputs.size()
        subsequent_mask = torch.triu(torch.ones((seq_length, seq_length), device=inputs.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # BxTxT

        return subsequent_mask.bool()

    return _get_attn_key_pad_mask(seq_k, seq_q, pad_id) | _get_subsequent_mask(seq_k)


def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """ mask position is set to 1 """
    non_pad_mask = get_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask
