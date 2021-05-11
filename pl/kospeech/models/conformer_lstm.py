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
from omegaconf import DictConfig

from kospeech.metrics import WordErrorRate, CharacterErrorRate
from kospeech.models.decoder.lstm import DecoderLSTM
from kospeech.models.encoder.conformer import ConformerEncoder
from kospeech.models.kospeech_model import KospeechEncoderDecoderModel
from kospeech.vocabs import KsponSpeechVocabulary
from kospeech.vocabs.vocab import Vocabulary


class ConformerLSTMModel(KospeechEncoderDecoderModel):
    """
    PyTorch Lightning Automatic Speech Recognition Model. It consist of a conformer encoder and rnn decoder.

    Args:
        configs (DictConfig): configuraion set
        num_classes (int): number of classification classes
        vocab (Vocabulary): vocab of training data
        wer_metric (WordErrorRate): metric for measuring speech-to-text accuracy of ASR systems (word-level)
        cer_metric (CharacterErrorRate): metric for measuring speech-to-text accuracy of ASR systems (character-level)
        criterion_name (str): name of criterion for loss computing
    """
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = KsponSpeechVocabulary,
            wer_metric: WordErrorRate = WordErrorRate,
            cer_metric: CharacterErrorRate = CharacterErrorRate,
            criterion_name: str = 'joint_ctc_cross_entropy',
    ) -> None:
        super(ConformerLSTMModel, self).__init__(configs, num_classes, vocab, wer_metric, cer_metric, criterion_name)

    def build_encoder(self, configs: DictConfig, num_classes: int):
        self.encoder = ConformerEncoder(
            input_dim=configs.num_mels,
            encoder_dim=configs.encoder_dim,
            num_layers=configs.num_encoder_layers,
            num_attention_heads=configs.num_attention_heads,
            feed_forward_expansion_factor=configs.feed_forward_expansion_factor,
            conv_expansion_factor=configs.conv_expansion_factor,
            input_dropout_p=configs.input_dropout_p,
            feed_forward_dropout_p=configs.feed_forward_dropout_p,
            attention_dropout_p=configs.attention_dropout_p,
            conv_dropout_p=configs.conv_dropout_p,
            conv_kernel_size=configs.conv_kernel_size,
            half_step_residual=configs.half_step_residual,
        )

    def build_decoder(self, configs: DictConfig, num_classes: int):
        self.decoder = DecoderLSTM(
            num_classes=num_classes,
            max_length=configs.max_length,
            hidden_state_dim=configs.encoder_dim,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            num_heads=configs.num_attention_heads,
            dropout_p=configs.decoder_dropout_p,
            num_layers=configs.num_decoder_layers,
            rnn_type=configs.rnn_type,
            use_tpu=configs.use_tpu,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        return super(ConformerLSTMModel, self).forward(inputs, input_lengths)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        return super(ConformerLSTMModel, self).training_step(batch, batch_idx)

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        return super(ConformerLSTMModel, self).validation_step(batch, batch_idx)

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        return super(ConformerLSTMModel, self).test_step(batch, batch_idx)
