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

from kospeech.models.interface import TransducerInterface


class RNNTransducder(TransducerInterface):
    def __init__(self, encoder, decoder, joint_net):
        super(RNNTransducder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint_net = joint_net

    def forward(
            self,
            inputs: Tensor,
            inputs_lengths: Tensor,
            targets: Tensor,
            targets_lengths: Tensor
    ) -> Tensor:
        encoder_outputs = self.encoder(inputs, inputs_lengths)
        decoder_outputs = self.decoder(targets, targets_lengths)
        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs
