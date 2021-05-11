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
import random
from torch import Tensor
from typing import Optional

from kospeech.models.decoder.base import IncrementalDecoder
from kospeech.models.modules.modules import LayerNorm, Linear
from kospeech.models.modules.transformer.decoder import TransformerDecoderLayer
from kospeech.models.modules.transformer.embeddings import Embedding, PositionalEncoding
from kospeech.models.modules.transformer.mask import get_decoder_self_attn_mask, get_attn_pad_mask


class TransformerDecoder(IncrementalDecoder):
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
            num_classes: int,
            d_model: int = 512,
            d_ff: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout_p: float = 0.3,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            max_length: int = 128,
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
            LayerNorm(d_model),
            Linear(d_model, num_classes, bias=False),
        )

    def forward(
            self,
            encoder_outputs: Tensor = None,
            targets: Optional[Tensor] = None,
            encoder_output_lengths: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths: The length of encoder outputs. ``(batch)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        batch_size = encoder_outputs.size(0)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if targets is not None and use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            target_length = targets.size(1)

            self_attn_mask = get_decoder_self_attn_mask(targets, targets, self.pad_id)
            encoder_outputs_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, target_length)

            outputs = self.embedding(targets) + self.positional_encoding(target_length)
            outputs = self.input_dropout(outputs)

            for layer in self.layers:
                outputs, self_attn, memory_attn = layer(
                    inputs=outputs,
                    encoder_outputs=encoder_outputs,
                    self_attn_mask=self_attn_mask,
                    encoder_outputs_mask=encoder_outputs_mask,
                )

            predicted_log_probs = self.fc(outputs).log_softmax(dim=-1)

        else:  # Inference
            predicted_log_probs = encoder_outputs.new_zeros(batch_size, self.max_length).long()
            predicted_log_probs[:, 0] = self.sos_id

            for di in range(1, self.max_length):
                step_outputs = self.forward(predicted_log_probs, encoder_outputs, encoder_output_lengths)
                step_outputs = step_outputs.max(dim=-1, keepdim=False)[1]
                predicted_log_probs[:, di] = step_outputs[:, di]

        return predicted_log_probs
