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

import pdb
from torch import Tensor
from typing import Tuple

from kospeech.models.interface import EncoderDecoderModelInterface
from kospeech.models.transformer.decoder import TransformerDecoder
from kospeech.models.transformer.encoder import TransformerEncoder


class SpeechTransformer(EncoderDecoderModelInterface):
    """
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        encoder (TransformerEncoder): encoder of transformer
        decoder (TransformerDecoder): decoder of transformer
        num_classes (int): the number of classfication
        d_model (int): dimension of model (default: 512)
        pad_id (int): identification of <PAD_token>
        eos_id (int): identification of <EOS_token>
        num_heads (int): number of attention heads (default: 8)

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
            encoder: TransformerEncoder,            # encoder of transformer
            decoder: TransformerDecoder,            # decoder of transformer
            num_classes: int,                       # the number of classfication
            d_model: int = 512,                     # dimension of model
            pad_id: int = 0,                        # identification of <PAD_token>
            sos_id: int = 1,                        # identification of <SOS_token>
            eos_id: int = 2,                        # identification of <EOS_token>
            num_heads: int = 8,                     # number of attention heads
            joint_ctc_attention: bool = False,      # flag indication whether to apply joint ctc attention
            max_length: int = 400,                   # a maximum allowed length for the sequence to be processed
    ) -> None:
        super(SpeechTransformer, self).__init__(encoder, decoder)
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_classes = num_classes
        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        encoder_outputs, output_lengths, encoder_log_probs = self.encoder(inputs, input_lengths)
        predicted_log_probs = self.decoder(targets, encoder_outputs, output_lengths)
        return predicted_log_probs, encoder_log_probs, output_lengths
