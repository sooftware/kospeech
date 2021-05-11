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

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Union, Optional, Tuple
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Adam, Adagrad, Adadelta, Adamax, SGD, ASGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kospeech.criterion import JointCTCCrossEntropyLoss, LabelSmoothedCrossEntropyLoss, TransducerLoss
from kospeech.metrics import WordErrorRate, CharacterErrorRate
from kospeech.optim import AdamP, RAdam
from kospeech.optim.lr_scheduler import TriStageLRScheduler, TransformerLRScheduler
from kospeech.vocabs import KsponSpeechVocabulary
from kospeech.vocabs.vocab import Vocabulary


class BaseKospeechModel(pl.LightningModule):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = KsponSpeechVocabulary,
            wer_metric: WordErrorRate = WordErrorRate,
            cer_metric: CharacterErrorRate = CharacterErrorRate,
    ) -> None:
        super(BaseKospeechModel, self).__init__()
        self.configs = configs
        self.num_classes = num_classes
        self.gradient_clip_val = configs.gradient_clip_val
        self.vocab = vocab
        self.wer_metric = wer_metric
        self.cer_metric = cer_metric

    def _log_states(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch: tuple, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch: tuple, batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, object, str]]:
        supported_optimizers = {
            "adam": Adam,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
        }
        assert self.configs.optimizer in supported_optimizers.keys(), \
            f"Unsupported Optimizer: {self.configs.optimizer}\n" \
            f"Supported Optimizers: {supported_optimizers.keys()}"
        optimizer = supported_optimizers[self.configs.optimizer](self.parameters(), lr=self.configs.lr)

        if self.configs.scheduler == 'transformer':
            scheduler = TransformerLRScheduler(
                optimizer,
                peak_lr=self.configs.peak_lr,
                final_lr=self.configs.final_lr,
                final_lr_scale=self.configs.final_lr_scale,
                warmup_steps=self.configs.warmup_steps,
                decay_steps=self.configs.decay_steps,
            )
        elif self.configs.scheduler == 'tri_stage':
            scheduler = TriStageLRScheduler(
                optimizer,
                init_lr=self.configs.init_lr,
                peak_lr=self.configs.peak_lr,
                final_lr=self.configs.final_lr,
                final_lr_scale=self.configs.final_lr_scale,
                init_lr_scale=self.configs.init_lr_scale,
                warmup_steps=self.configs.warmup_steps,
                total_steps=self.configs.warmup_steps + self.configs.decay_steps,
            )
        elif self.configs.scheduler == 'reduce_lr_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.configs.lr_patience,
                factor=self.configs.lr_factor,
            )
        else:
            raise ValueError(f"Unsupported `scheduler`: {self.configs.scheduler}\n"
                             f"Supported `scheduler`: transformer, tri_stage, reduce_lr_on_plateau")

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'metric_to_track',
        }

    def configure_criterion(self, *args, **kwargs):
        raise NotImplementedError


class KospeechEncoderDecoderModel(BaseKospeechModel):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = KsponSpeechVocabulary,
            wer_metric: WordErrorRate = WordErrorRate,
            cer_metric: CharacterErrorRate = CharacterErrorRate,
            criterion_name: str = 'joint_ctc_cross_entropy',
    ) -> None:
        super(KospeechEncoderDecoderModel, self).__init__(configs, num_classes, vocab, wer_metric, cer_metric)
        self.teacher_forcing_ratio = configs.teacher_forcing_ratio
        self.encoder = None
        self.decoder = None
        self.criterion = self.configure_criterion(criterion_name)

    def _log_states(
            self,
            stage: str,
            wer: float,
            cer: float,
            loss: float,
            cross_entropy_loss: Optional[float] = None,
            ctc_loss: Optional[float] = None,
    ) -> None:
        self.log(f"{stage}_wer", wer)
        self.log(f"{stage}_cer", cer)
        self.log(f"{stage}_loss", loss)
        if cross_entropy_loss is not None:
            self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        if ctc_loss is not None:
            self.log(f"{stage}_ctc_loss", ctc_loss)
        return

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_results(
            self,
            stage: str,
            y_hats: Tensor,
            encoder_log_probs: Tensor,
            encoder_output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        if isinstance(self.criterion, JointCTCCrossEntropyLoss):
            loss, ctc_loss, cross_entropy_loss = self.criterion(
                encoder_log_probs=encoder_log_probs.transpose(0, 1),
                decoder_log_probs=y_hats.contiguous().view(-1, y_hats.size(-1)),
                output_lengths=encoder_output_lengths,
                targets=targets[:, 1:],
                target_lengths=target_lengths,
            )
            wer = self.wer_metric(targets[:, 1:], y_hats)
            cer = self.cer_metric(targets[:, 1:], y_hats)

            self._log_states(stage, wer, cer, loss, cross_entropy_loss, ctc_loss)

        elif isinstance(self.criterion, LabelSmoothedCrossEntropyLoss):
            loss = self.criterion(
                y_hats.contiguous().view(-1, y_hats.size(-1)),
                targets[:, 1:].contiguous().view(-1),
            )
            wer = self.wer_metric(targets[:, 1:], y_hats)
            cer = self.cer_metric(targets[:, 1:], y_hats)

            self._log_states(stage, wer, cer, loss)

        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}"
                             f"Supported criterion: `JointCTCCrossEntropyLoss`, `LabelSmoothedCrossEntropyLoss`")

        return loss

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for inference.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * y_hats (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, _, encoder_output_lengths = self.encoder(inputs, input_lengths)
        return self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_log_probs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs, targets, encoder_output_lengths, self.teacher_forcing_ratio)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = y_hats[:, :max_target_length, :]

        return self._compute_results(
            stage='train',
            y_hats=y_hats,
            encoder_log_probs=encoder_log_probs,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_log_probs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs, encoder_output_lengths=encoder_output_lengths, teacher_forcing_ratio=0.0)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = y_hats[:, :max_target_length, :]

        return self._compute_results(
            stage='valid',
            y_hats=y_hats,
            encoder_log_probs=encoder_log_probs,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_log_probs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs, encoder_output_lengths=encoder_output_lengths, teacher_forcing_ratio=0.0)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = y_hats[:, :max_target_length, :]

        return self._compute_results(
            stage='test',
            y_hats=y_hats,
            encoder_log_probs=encoder_log_probs,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def configure_criterion(self, criterion: str = 'joint_ctc_cross_entropy') -> nn.Module:
        if criterion == 'joint_ctc_cross_entropy':
            criterion = JointCTCCrossEntropyLoss(
                num_classes=self.num_classes,
                ignore_index=self.vocab.pad_id,
                blank_id=self.vocab.blank_id,
                ctc_weight=self.configs.ctc_weight,
                cross_entropy_weight=self.configs.cross_entropy_weight,
            )
        elif criterion == 'label_smoothed_cross_entropy':
            criterion = LabelSmoothedCrossEntropyLoss(
                num_classes=self.num_classes,
                ignore_index=self.vocab.pad_id,
                smoothing=self.configs.smoothing,
            )
        elif criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id)

        return criterion


class KospeechEncoderModel(BaseKospeechModel):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = KsponSpeechVocabulary,
            wer_metric: WordErrorRate = WordErrorRate,
            cer_metric: CharacterErrorRate = CharacterErrorRate,
    ) -> None:
        super(KospeechEncoderModel, self).__init__(configs, num_classes, vocab, wer_metric, cer_metric)
        self.encoder = None
        self.decoder = None
        self.criterion = self.configure_criterion()

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def set_decoder(self, decoder: nn.Module):
        """ Set decoder for beam search """
        self.decoder = decoder

    def _compute_results(
            self,
            stage: str,
            y_hats: Tensor,
            output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        loss = self.criterion(
            y_hats.transpose(0, 1),
            targets[:, 1:],
            output_lengths,
            target_lengths,
        )
        wer = self.wer_metric(targets[:, 1:], y_hats)
        cer = self.cer_metric(targets[:, 1:], y_hats)

        self._log_states(stage, wer, cer, loss)

        return loss

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for inference.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * y_hats (torch.FloatTensor): Result of model predictions.
        """
        predicted_log_probs, output_lengths = self.encoder(inputs, input_lengths)
        if self.decoder is not None:
            y_hats = self.decoder(predicted_log_probs)
        else:
            y_hats = predicted_log_probs.max(-1)[1]
        return y_hats, output_lengths

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch
        y_hats, output_lengths = self.encoder(inputs, input_lengths)
        return self._compute_results(
            stage='train',
            y_hats=y_hats,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch
        y_hats, output_lengths = self.encoder(inputs, input_lengths)
        return self._compute_results(
            stage='valid',
            y_hats=y_hats,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch
        y_hats, output_lengths = self.encoder(inputs, input_lengths)
        return self._compute_results(
            stage='test',
            y_hats=y_hats,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def configure_criterion(self):
        return nn.CTCLoss(
            blank=self.vocab.blank_id,
            reduction="mean",
            zero_infinity=True,
        )


class KospeechTransducerModel(BaseKospeechModel):
    def __init__(
            self,
            configs: DictConfig,
            num_classes: int,
            vocab: Vocabulary = KsponSpeechVocabulary,
            wer_metric: WordErrorRate = WordErrorRate,
            cer_metric: CharacterErrorRate = CharacterErrorRate,
    ) -> None:
        super(KospeechTransducerModel, self).__init__(configs, num_classes, vocab, wer_metric, cer_metric)
        self.encoder = None
        self.decoder = None
        self.criterion = self.configure_criterion()

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_results(
            self,
            stage: str,
            log_probs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        loss = self.criterion(
            log_probs=log_probs,
            targets=targets[:, 1:].contiguous().int(),
            input_lengths=input_lengths.int(),
            target_lengths=target_lengths.int(),
        )
        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        y_hats = log_probs[:, :max_target_length, :]

        wer = self.wer_metric(targets[:, 1:], y_hats)
        cer = self.cer_metric(targets[:, 1:], y_hats)

        self._log_states(stage, wer, cer, loss)

        return loss

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs).log_softmax(dim=-1)

        return outputs

    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        """
        Decode `encoder_outputs`.

        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Decode `encoder_outputs`.


        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        return torch.stack(outputs, dim=1).transpose(0, 1)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        predicted_log_probs = self.joint(encoder_outputs, decoder_outputs)
        return self._compute_results(
            'train',
            log_probs=predicted_log_probs,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        outputs = list()
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        predicted_log_probs = torch.stack(outputs, dim=1).transpose(0, 1)
        return self._compute_results(
            'valid',
            log_probs=predicted_log_probs,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        outputs = list()
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        predicted_log_probs = torch.stack(outputs, dim=1).transpose(0, 1)
        return self._compute_results(
            'test',
            log_probs=predicted_log_probs,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def configure_criterion(self):
        return TransducerLoss(self.vocab.blank_id)
