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
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from kospeech.models.attention import MultiHeadAttention
from kospeech.models.decoder import BaseDecoder
from kospeech.models.modules import Linear
from kospeech.models.transformer.sublayers import PositionwiseFeedForward
from kospeech.models.transformer.embeddings import Embedding, PositionalEncoding
from kospeech.models.transformer.mask import get_attn_pad_mask, get_attn_subsequent_mask


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_prenorm = nn.LayerNorm(d_model)
        self.encoder_attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Optional[Tensor] = None,
            encoder_outputs_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.encoder_attention_prenorm(outputs)
        outputs, encoder_attn = self.encoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_outputs_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, self_attn, encoder_attn


class TransformerDecoder(BaseDecoder):
    """
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        num_classes: umber of classes
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of decoder layers
        num_heads: number of attention heads
        dropout_p: probability of dropout
        pad_id: identification of pad token
        eos_id: identification of end of sentence token
    """

    def __init__(
            self,
            num_classes: int,               # number of classes
            d_model: int = 512,             # dimension of model
            d_ff: int = 512,                # dimension of feed forward network
            num_layers: int = 6,            # number of decoder layers
            num_heads: int = 8,             # number of attention heads
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
            sos_id: int = 1,                # identification of start of sentence token
            eos_id: int = 2,                # identification of end of sentence token
            max_length: int = 400,          # max length of decoding
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, num_classes, bias=False),
        )

    def forward_step(
            self,
            decoder_inputs,
            decoder_input_lengths,
            encoder_outputs,
            encoder_output_lengths,
            positional_encoding_length,
    ) -> Tensor:
        dec_self_attn_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_input_lengths, decoder_inputs.size(1)
        )
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(
            encoder_outputs, encoder_output_lengths, decoder_inputs.size(1)
        )

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )

        return outputs

    def forward(
            self,
            targets: Tensor,
            encoder_outputs: Tensor,
            encoder_output_lengths: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths: The length of encoder outputs. ``(batch)``

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        batch_size = encoder_outputs.size(0)

        targets = targets[targets != self.eos_id].view(batch_size, -1)
        target_length = targets.size(1)

        outputs = self.forward_step(
            decoder_inputs=targets,
            decoder_input_lengths=target_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=target_length,
        )
        return self.fc(outputs).log_softmax(dim=-1)

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tensor:
        logits = list()
        batch_size = encoder_outputs.size(0)

        input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
        input_var = input_var.fill_(self.pad_id)
        input_var[:, 0] = self.sos_id

        for di in range(1, self.max_length):
            input_lengths = torch.IntTensor(batch_size).fill_(di)

            outputs = self.forward_step(
                decoder_inputs=input_var[:, :di],
                decoder_input_lengths=input_lengths,
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                positional_encoding_length=di,
            )
            step_output = self.fc(outputs).log_softmax(dim=-1)

            logits.append(step_output[:, -1, :])
            input_var = logits[-1].topk(1)[1]

        return torch.stack(logits, dim=1)
