# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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

from kospeech.models.model import TransducerModel
from kospeech.models.rnnt.decoder import DecoderRNNT
from kospeech.models.rnnt.encoder import EncoderRNNT


class RNNTransducer(TransducerModel):
    """
    RNN-Transducer are a form of sequence-to-sequence models that do not employ attention mechanisms.
    Unlike most sequence-to-sequence models, which typically need to process the entire input sequence
    (the waveform in our case) to produce an output (the sentence), the RNN-T continuously processes input samples and
    streams output symbols, a property that is welcome for speech dictation. In our implementation,
    the output symbols are the characters of the alphabet.

    Args:
        num_classes (int): number of classification
        input_dim (int): dimension of input vector
        num_encoder_layers (int, optional): number of encoder layers (default: 4)
        num_decoder_layers (int, optional): number of decoder layers (default: 1)
        encoder_hidden_state_dim (int, optional): hidden state dimension of encoder (default: 320)
        decoder_hidden_state_dim (int, optional): hidden state dimension of decoder (default: 512)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
        encoder_dropout_p (float, optional): dropout probability of encoder
        decoder_dropout_p (float, optional): dropout probability of decoder
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification

    Inputs: inputs, input_lengths, targets, target_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 1,
            encoder_hidden_state_dim: int = 320,
            decoder_hidden_state_dim: int = 512,
            output_dim: int = 512,
            rnn_type: str = "lstm",
            bidirectional: bool = True,
            encoder_dropout_p: float = 0.2,
            decoder_dropout_p: float = 0.2,
            sos_id: int = 1,
            eos_id: int = 2,
    ):
        encoder = EncoderRNNT(
            input_dim=input_dim,
            hidden_state_dim=encoder_hidden_state_dim,
            output_dim=output_dim,
            num_layers=num_encoder_layers,
            rnn_type=rnn_type,
            dropout_p=encoder_dropout_p,
            bidirectional=bidirectional,
        )
        decoder = DecoderRNNT(
            num_classes=num_classes,
            hidden_state_dim=decoder_hidden_state_dim,
            output_dim=output_dim,
            num_layers=num_decoder_layers,
            rnn_type=rnn_type,
            sos_id=sos_id,
            eos_id=eos_id,
            dropout_p=decoder_dropout_p,
        )
        super(RNNTransducer, self).__init__(encoder, decoder, output_dim, num_classes)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        return super().forward(inputs, input_lengths, targets, target_lengths)
