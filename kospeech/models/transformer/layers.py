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

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Any
from kospeech.models.attention import MultiHeadAttention
from kospeech.models.transformer.sublayers import (
    AddNorm,
    PositionWiseFeedForwardNet
)


class SpeechTransformerEncoderLayer(nn.Module):
    """
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            ffnet_style: str = 'ff'         # style of feed forward network
    ) -> None:
        super(SpeechTransformerEncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs: Tensor, self_attn_mask: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs = self.feed_forward(outputs)
        return outputs, attn


class SpeechTransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            ffnet_style: str = 'ff'         # style of feed forward network
    ) -> None:
        super(SpeechTransformerDecoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.memory_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(
            self,
            inputs: Tensor,                                 # B x T_input
            memory: Tensor,                                 # B x T_input x D_model
            self_attn_mask: Optional[Any] = None,           # B x T_input x T_input
            memory_mask: Optional[Any] = None               # B x T_input x T_output
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs, memory_attn = self.memory_attention(outputs, memory, memory, memory_mask)
        outputs = self.feed_forward(outputs)
        return outputs, self_attn, memory_attn
