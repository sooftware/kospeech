# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Any
from kospeech.models.transformer.sublayers import AddNorm, PositionWiseFeedForwardNet
from kospeech.models.attention import MultiHeadAttention


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

    def forward(
            self,
            inputs: Tensor,                             # B x T_input x D
            non_pad_mask: Optional[Any] = None,         # B x T_input
            self_attn_mask: Optional[Any] = None        # B x T_input x T_output
    ) -> Tuple[Tensor, Tensor]:
        output, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output *= non_pad_mask

        output = self.feed_forward(output)
        output *= non_pad_mask

        return output, attn


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
            non_pad_mask: Optional[Any] = None,             # B x T_input
            self_attn_mask: Optional[Any] = None,           # B x T_input x T_input
            memory_mask: Optional[Any] = None               # B x T_input x T_output
    ) -> Tuple[Tensor, Tensor, Tensor]:
        output, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output *= non_pad_mask

        output, memory_attn = self.memory_attention(output, memory, memory, memory_mask)
        output *= non_pad_mask

        output = self.feed_forward(output)
        output *= non_pad_mask

        return output, self_attn, memory_attn
