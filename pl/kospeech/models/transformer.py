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

from omegaconf import DictConfig
from torch import Tensor

from kospeech.metrics import WordErrorRate, CharacterErrorRate
from kospeech.models.decoder.transformer import TransformerDecoder
from kospeech.models.encoder.transformer import TransformerEncoder
from kospeech.models.kospeech_model import KospeechEncoderDecoderModel
from kospeech.vocabs import KsponSpeechVocabulary
from kospeech.vocabs.vocab import Vocabulary


class SpeechTransformerModel(KospeechEncoderDecoderModel):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = KsponSpeechVocabulary,
            wer_metric: WordErrorRate = WordErrorRate,
            cer_metric: CharacterErrorRate = CharacterErrorRate,
            criterion_name: str = 'joint_ctc_cross_entropy',
    ) -> None:
        super(SpeechTransformerModel, self).__init__(configs, num_classes, vocab,
                                                     wer_metric, cer_metric, criterion_name)

    def build_encoder(self, configs: DictConfig, num_classes: int):
        self.encoder = TransformerEncoder(
            input_dim=configs.input_dim,
            extractor=configs.extractor,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            num_layers=configs.num_layers,
            num_heads=configs.num_heads,
            dropout_p=configs.dropout_p,
            joint_ctc_attention=configs.joint_ctc_attention,
            num_classes=num_classes,
        )

    def build_decoder(self, configs: DictConfig, num_classes: int):
        self.decoder = TransformerDecoder(
            num_classes=num_classes,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            num_layers=configs.num_layers,
            num_heads=configs.num_heads,
            dropout_p=configs.dropout_p,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            max_length=configs.max_length,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        return super(SpeechTransformerModel, self).forward(inputs, input_lengths)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        return super(SpeechTransformerModel, self).training_step(batch, batch_idx)

    def validation_step(self, batch: tuple, batch_idx: int, dataset_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        return super(SpeechTransformerModel, self).validation_step(batch, batch_idx)

    def test_step(self, batch: tuple, batch_idx: int, dataset_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        return super(SpeechTransformerModel, self).test_step(batch, batch_idx)
