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
from typing import Tuple
from kospeech.models.conv import (
    VGGExtractor,
    DeepSpeech2Extractor,
    Conv2dSubsampling,
)
from kospeech.models.model import EncoderDecoderModel
from kospeech.models.transformer.decoder import SpeechTransformerDecoder
from kospeech.models.transformer.encoder import SpeechTransformerEncoder


class SpeechTransformer(EncoderDecoderModel):
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
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        if extractor.lower() == 'vgg':
            conv = VGGExtractor(input_dim)
        elif extractor.lower() == 'ds2':
            conv = DeepSpeech2Extractor(input_dim)
        elif extractor.lower() == 'conv2d':
            conv = Conv2dSubsampling(input_dim, in_channels=1, out_channels=d_model)
        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

        super(SpeechTransformer, self).__init__(
            encoder=SpeechTransformerEncoder(
                num_classes=num_classes,
                d_model=d_model,
                input_dim=conv.get_output_dim(),
                d_ff=d_ff,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                ffnet_style=ffnet_style,
                dropout_p=dropout_p,
                pad_id=pad_id,
                joint_ctc_attention=joint_ctc_attention,
            ),
            decoder=SpeechTransformerDecoder(
                num_classes=num_classes,
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_decoder_layers,
                num_heads=num_heads,
                ffnet_style=ffnet_style,
                dropout_p=dropout_p,
                pad_id=pad_id,
                eos_id=eos_id,
            ),
        )

        self.num_classes = num_classes
        self.extractor = extractor.lower()
        self.joint_ctc_attention = joint_ctc_attention
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length
        self.conv = conv

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        outputs, output_lengths = self.conv(inputs, input_lengths)
        encoder_outputs, encoder_log_probs, output_lengths = self.encoder(outputs, output_lengths)
        predicted_log_probs = self.decoder(targets, input_lengths, encoder_outputs)
        return predicted_log_probs, encoder_log_probs, output_lengths

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        batch_size = inputs.size(0)

        outputs, output_lengths = self.conv(inputs, input_lengths)
        memory, _, _ = self.encoder(outputs, output_lengths)
        y_hats = memory.new_zeros(batch_size, self.max_length).long()
        y_hats[:, 0] = self.sos_id

        for di in range(1, self.max_length):
            step_outputs = self.decoder(y_hats, input_lengths, memory)
            step_outputs = step_outputs.max(dim=-1, keepdim=False)[1]
            y_hats[:, di] = step_outputs[:, di]

        return y_hats
