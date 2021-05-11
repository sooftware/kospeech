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
from typing import Tuple

from kospeech.models.encoder.base import BaseEncoder
from kospeech.models.modules.transformer.embeddings import PositionalEncoding
from kospeech.models.modules.transformer.encoder import TransformerEncoderLayer
from kospeech.models.modules.transformer.mask import get_attn_pad_mask
from kospeech.models.modules.modules import Linear, LayerNorm


class TransformerEncoder(BaseEncoder):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        input_dim: dimension of feature vector
        extractor (str): convolutional extractor
        d_model: dimension of model (default: 512)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoder layers (default: 6)
        num_heads: number of attention heads (default: 8)
        dropout_p:  probability of dropout (default: 0.3)

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths
    """

    def __init__(
            self,
            input_dim: int,
            extractor: str = 'vgg',
            d_model: int = 512,
            d_ff: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout_p: float = 0.3,
            joint_ctc_attention: bool = False,
            num_classes: int = None,
    ) -> None:
        super(TransformerEncoder, self).__init__(input_dim=input_dim, extractor=extractor, d_model=d_model,
                                                 num_classes=num_classes, dropout_p=dropout_p,
                                                 joint_ctc_attention=joint_ctc_attention)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = Linear(self.conv_output_dim, d_model)
        self.input_layer_norm = LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor, Tensor):

            * outputs: A output sequence of encoder. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * output_lengths: The length of encoder outputs. ``(batch)``
            * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        """
        encoder_log_probs = None

        features, output_lengths = self.conv(inputs, input_lengths)
        self_attn_mask = get_attn_pad_mask(features, output_lengths, features.size(1))

        outputs = self.input_layer_norm(self.input_proj(features)) + self.positional_encoding(features.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        if self.joint_ctc_attention:
            encoder_log_probs = self.fc(outputs.transpose(1, 2)).log_softmax(dim=-1)

        return outputs, encoder_log_probs, output_lengths
