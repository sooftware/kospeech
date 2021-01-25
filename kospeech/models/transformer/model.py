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

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import (
    Optional,
    Any,
    Tuple,
    Union,
)
from kospeech.models.conv import (
    VGGExtractor,
    DeepSpeech2Extractor,
)
from kospeech.models.modules import (
    Linear,
    LayerNorm, 
    Transpose,
)
from kospeech.models.transformer.mask import (
    get_attn_pad_mask,
    get_decoder_self_attn_mask,
)
from kospeech.models.transformer.embeddings import (
    Embedding,
    PositionalEncoding,
)
from kospeech.models.transformer.layers import (
    SpeechTransformerEncoderLayer,
    SpeechTransformerDecoderLayer,
)


class SpeechTransformer(nn.Module):
    """
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        num_classes (int): the number of classfication
        d_model (int): dimension of model (default: 512)
        input_dim (int): dimension of input
        pad_id (int): identification of <PAD_token>
        eos_id (int): identification of <EOS_token>
        d_ff (int): dimension of feed forward network (default: 2048)
        num_encoder_layers (int): number of encoder layers (default: 6)
        num_decoder_layers (int): number of decoder layers (default: 6)
        num_heads (int): number of attention heads (default: 8)
        dropout_p (float): dropout probability (default: 0.3)
        ffnet_style (str): if poswise_ffnet is 'ff', position-wise feed forware network to be a feed forward,
            otherwise, position-wise feed forward network to be a convolution layer. (default: ff)

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.

    Returns: output
        - **output**: tensor containing the outputs
    """

    def __init__(
            self,
            num_classes: int,                       # the number of classfication
            d_model: int = 512,                     # dimension of model
            input_dim: int = 80,                    # dimension of input
            pad_id: int = 0,                        # identification of <PAD_token>
            sos_id: int = 1,                        # identification of <SOS_token>
            eos_id: int = 2,                        # identification of <EOS_token>
            d_ff: int = 2048,                       # dimension of feed forward network
            num_heads: int = 8,                     # number of attention heads
            num_encoder_layers: int = 6,            # number of encoder layers
            num_decoder_layers: int = 6,            # number of decoder layers
            dropout_p: float = 0.3,                 # dropout probability
            ffnet_style: str = 'ff',                # feed forward network style 'ff' or 'conv'
            extractor: str = 'vgg',                 # CNN extractor [vgg, ds2]
            joint_ctc_attention: bool = False,      # flag indication whether to apply joint ctc attention
            max_length: int = 400,                  # a maximum allowed length for the sequence to be processed
    ) -> None:
        super(SpeechTransformer, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_classes = num_classes
        self.extractor = extractor
        self.joint_ctc_attention = joint_ctc_attention
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

        if self.extractor == 'vgg':
            input_dim = (input_dim - 1) << 5 if input_dim % 2 else input_dim << 5
            self.conv = VGGExtractor(mask_conv=False)

        elif self.extractor == 'ds2':
            input_dim = int(math.floor(input_dim + 2 * 20 - 41) / 2 + 1)
            input_dim = int(math.floor(input_dim + 2 * 10 - 21) / 2 + 1)
            input_dim <<= 6
            self.conv = DeepSpeech2Extractor(mask_conv=False)

        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

        self.encoder = SpeechTransformerEncoder(
            d_model=d_model,
            input_dim=input_dim,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id,
        )

        if self.joint_ctc_attention:
            self.encoder_fc = nn.Sequential(
                nn.BatchNorm1d(d_model),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(d_model, num_classes, bias=False),
            )

        self.decoder = SpeechTransformerDecoder(
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ffnet_style=ffnet_style,
            dropout_p=dropout_p,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        self.decoder_fc = Linear(d_model, num_classes)

    def forward(
            self,
            inputs: Tensor,                         # tensor of input sequences
            input_lengths: Tensor,                  # tensor of input sequence lengths
            targets: Optional[Tensor] = None,       # tensor of target sequences
    ) -> Union[Tensor, tuple]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        encoder_log_probs = None

        conv_feat = self.conv(inputs.unsqueeze(1), input_lengths)
        conv_feat = conv_feat.transpose(1, 2)

        batch_size, seq_length, num_channels, hidden_dim = conv_feat.size()
        conv_feat = conv_feat.contiguous().view(batch_size, seq_length, num_channels * hidden_dim)

        if self.extractor == 'vgg':
            input_lengths = (input_lengths >> 2).int()

        memory = self.encoder(conv_feat, input_lengths)
        if self.joint_ctc_attention:
            encoder_log_probs = self.encoder_fc(memory.transpose(1, 2)).log_softmax(dim=2)

        outputs = self.decoder(targets, input_lengths, memory)
        outputs = self.decoder_fc(outputs)

        return outputs, encoder_log_probs, input_lengths

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: str):
        with torch.no_grad():
            conv_feat = self.conv(inputs.unsqueeze(1), input_lengths)
            conv_feat = conv_feat.transpose(1, 2)

            batch_size, seq_length, num_channels, hidden_dim = conv_feat.size()
            conv_feat = conv_feat.contiguous().view(batch_size, seq_length, num_channels * hidden_dim)

            if self.extractor == 'vgg':
                input_lengths = (input_lengths >> 2).int()

            memory = self.encoder(conv_feat, input_lengths)
            y_hats = memory.new_zeros(batch_size, self.max_length).long()
            y_hats[:, 0] = self.sos_id

            for di in range(1, self.max_length):
                step_output = self.decoder(y_hats, input_lengths, memory)
                step_output = self.decoder_fc(step_output)
                step_output = step_output.max(dim=-1, keepdim=False)[1]
                y_hats[:, di] = step_output[:, di]

        return y_hats


class SpeechTransformerEncoder(nn.Module):
    """
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        d_model: dimension of model (default: 512)
        input_dim: dimension of feature vector (default: 80)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoder layers (default: 6)
        num_heads: number of attention heads (default: 8)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
        dropout_p:  probability of dropout (default: 0.3)
        pad_id: identification of pad token (default: 0)

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            input_dim: int = 80,            # dimension of feature vector
            d_ff: int = 2048,               # dimension of feed forward network
            num_layers: int = 6,            # number of encoder layers
            num_heads: int = 8,             # number of attention heads
            ffnet_style: str = 'ff',        # style of feed forward network [ff, conv]
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
    ) -> None:
        super(SpeechTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.input_proj = Linear(input_dim, d_model)
        self.input_layer_norm = LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [SpeechTransformerEncoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)]
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor = None) -> Tuple[Tensor, list]:
        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))

        outputs = self.input_layer_norm(self.input_proj(inputs)) + self.positional_encoding(inputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        return outputs


class SpeechTransformerDecoder(nn.Module):
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
        ffnet_style: style of feed forward network
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
            ffnet_style: str = 'ff',        # style of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
            eos_id: int = 2,                # identification of end of sentence token
    ) -> None:
        super(SpeechTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            SpeechTransformerDecoderLayer(d_model, num_heads, d_ff, dropout_p, ffnet_style) for _ in range(num_layers)
        ])
        self.pad_id = pad_id
        self.eos_id = eos_id

    def forward(self, inputs: Tensor, input_lengths: Optional[Any] = None, memory: Tensor = None):
        batch_size, output_length = inputs.size(0), inputs.size(1)

        self_attn_mask = get_decoder_self_attn_mask(inputs, inputs, self.pad_id)
        memory_mask = get_attn_pad_mask(memory, input_lengths, output_length)

        outputs = self.embedding(inputs) + self.positional_encoding(output_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(outputs, memory, self_attn_mask, memory_mask)

        return outputs
