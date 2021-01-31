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

from kospeech.models.model import TransducerModel
from kospeech.models.rnnt.decoder import DecoderRNNT
from kospeech.models.rnnt.encoder import EncoderRNNT


class RNNTransducer(TransducerModel):
    def __init__(
            self,
            num_classes: int,
            input_dim: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            encoder_hidden_state_dim: int,
            decoder_hidden_state_dim: int,
            output_dim: int,
            rnn_type: str,
            bidirectional: bool,
            encoder_dropout_p: float,
            decoder_dropout_p: float,
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
