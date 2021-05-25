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

from torch import Tensor
from typing import Tuple

from kospeech.models.model import EncoderDecoderModel
from kospeech.models.transformer.decoder import TransformerDecoder
from kospeech.models.transformer.encoder import TransformerEncoder


class SpeechTransformer(EncoderDecoderModel):
    """
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classification
        extractor (str): type of CNN extractor (default: vgg)
        num_encoder_layers (int, optional): number of recurrent layers (default: 12)
        num_decoder_layers (int, optional): number of recurrent layers (default: 6)
        encoder_dropout_p (float, optional): dropout probability of encoder (default: 0.2)
        decoder_dropout_p (float, optional): dropout probability of decoder (default: 0.2)
        d_model (int): dimension of model (default: 512)
        d_ff (int): dimension of feed forward net (default: 2048)
        pad_id (int): identification of <PAD_token> (default: 0)
        sos_id (int): identification of <SOS_token> (default: 1)
        eos_id (int): identification of <EOS_token> (default: 2)
        num_heads (int): number of attention heads (default: 8)
        max_length (int, optional): max decoding step (default: 400)
        joint_ctc_attention (bool, optional): flag indication joint ctc attention or not (default: False)

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.

    Returns:
        (Tensor, Tensor, Tensor)

        * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        * encoder_output_lengths: The length of encoder outputs. ``(batch)``
        * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
            If joint_ctc_attention is False, return None.
    """

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            extractor: str,
            num_encoder_layers: int = 12,
            num_decoder_layers: int = 6,
            encoder_dropout_p: float = 0.2,
            decoder_dropout_p: float = 0.2,
            d_model: int = 512,
            d_ff: int = 2048,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 8,
            joint_ctc_attention: bool = False,
            max_length: int = 400,
    ) -> None:
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        encoder = TransformerEncoder(
            input_dim=input_dim,
            extractor=extractor,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout_p=encoder_dropout_p,
            joint_ctc_attention=joint_ctc_attention,
            num_classes=num_classes,
        )
        decoder = TransformerDecoder(
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout_p=decoder_dropout_p,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            max_length=max_length,
        )
        super(SpeechTransformer, self).__init__(encoder, decoder)

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
            target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``

        Returns:
            (Tensor, Tensor, Tensor)

            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
            * encoder_output_lengths: The length of encoder outputs. ``(batch)``
            * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        """
        encoder_outputs, output_lengths, encoder_log_probs = self.encoder(inputs, input_lengths)
        predicted_log_probs = self.decoder(targets, encoder_outputs, output_lengths, target_lengths)
        return predicted_log_probs, output_lengths, encoder_log_probs
