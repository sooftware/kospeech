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

import os
import random
import warnings
import torch
import hydra

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from kospeech.data.data_loader import split_dataset
from kospeech.optim import Optimizer
from kospeech.optim.lr_scheduler import TriStageLRScheduler
from kospeech.trainer import SupervisedTrainer
from kospeech.model_builder import build_model
from kospeech.utils import (
    check_envirionment,
    get_optimizer,
    get_criterion,
    logger,
)
from kospeech.vocabs import (
    KsponSpeechVocabulary,
    LibriSpeechVocabulary,
)
from kospeech.data.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from kospeech.models import (
    DeepSpeech2Config,
    JointCTCAttentionLASConfig,
    ListenAttendSpellConfig,
    TransformerConfig,
    JointCTCAttentionTransformerConfig,
    JasperConfig,
)
from kospeech.trainer import (
    DeepSpeech2TrainConfig,
    ListenAttendSpellTrainConfig,
    TransformerTrainConfig,
    JasperTrainConfig,
)


def train(config: DictConfig):
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    device = check_envirionment(config.train.use_cuda)

    if config.train.dataset == 'kspon':
        if config.train.output_unit == 'subword':
            vocab = KsponSpeechVocabulary(
                vocab_path='../../../data/vocab/kspon_sentencepiece.vocab',
                output_unit=config.train.output_unit,
                sp_model_path='../../../data/vocab/kspon_sentencepiece.model',
            )
        else:
            vocab = KsponSpeechVocabulary(
                f'../../../data/vocab/aihub_{config.train.output_unit}_vocabs.csv', output_unit=config.train.output_unit
            )

    elif config.train.dataset == 'libri':
        vocab = LibriSpeechVocabulary(
            '../../../data/vocab/tokenizer.vocab', '../../../data/vocab/tokenizer.model'
        )

    else:
        raise ValueError("Unsupported Dataset : {0}".format(config.train.dataset))

    if not config.train.resume:
        epoch_time_step, trainset_list, validset = split_dataset(config, config.train.transcripts_path, vocab)
        model = build_model(config, vocab, device)

        optimizer = get_optimizer(model, config)

        lr_scheduler = TriStageLRScheduler(
            optimizer=optimizer,
            init_lr=config.train.init_lr,
            peak_lr=config.train.peak_lr,
            final_lr=config.train.final_lr,
            init_lr_scale=config.train.init_lr_scale,
            final_lr_scale=config.train.final_lr_scale,
            warmup_steps=config.train.warmup_steps,
            total_steps=int(config.train.num_epochs * epoch_time_step)
        )
        optimizer = Optimizer(optimizer, lr_scheduler, config.train.warmup_steps, config.train.max_grad_norm)
        criterion = get_criterion(config, vocab)

    else:
        trainset_list = None
        validset = None
        model = None
        optimizer = None
        epoch_time_step = None
        criterion = get_criterion(config, vocab)

    trainer = SupervisedTrainer(
        optimizer=optimizer,
        criterion=criterion,
        trainset_list=trainset_list,
        validset=validset,
        num_workers=config.train.num_workers,
        device=device,
        teacher_forcing_step=config.model.teacher_forcing_step,
        min_teacher_forcing_ratio=config.model.min_teacher_forcing_ratio,
        print_every=config.train.print_every,
        save_result_every=config.train.save_result_every,
        checkpoint_every=config.train.checkpoint_every,
        architecture=config.model.architecture,
        vocab=vocab,
        joint_ctc_attention=config.model.joint_ctc_attention,
    )
    model = trainer.train(
        model=model,
        batch_size=config.train.batch_size,
        epoch_time_step=epoch_time_step,
        num_epochs=config.train.num_epochs,
        teacher_forcing_ratio=config.model.teacher_forcing_ratio,
        resume=config.train.resume,
    )
    return model


cs = ConfigStore.instance()
cs.store(group="audio", name="fbank", node=FilterBankConfig, package="audio")
cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfig, package="audio")
cs.store(group="audio", name="mfcc", node=MfccConfig, package="audio")
cs.store(group="audio", name="spectrogram", node=SpectrogramConfig, package="audio")
cs.store(group="train", name="ds2_train", node=DeepSpeech2TrainConfig, package="train")
cs.store(group="train", name="las_train", node=ListenAttendSpellTrainConfig, package="train")
cs.store(group="train", name="transformer_train", node=TransformerTrainConfig, package="train")
cs.store(group="train", name="jasper_train", node=JasperTrainConfig, package="train")
cs.store(group="model", name="ds2", node=DeepSpeech2Config, package="model")
cs.store(group="model", name="las", node=ListenAttendSpellConfig, package="model")
cs.store(group="model", name="transformer", node=TransformerConfig, package="model")
cs.store(group="model", name="jasper", node=JasperConfig, package="model")
cs.store(group="model", name="joint-ctc-attention-las", node=JointCTCAttentionLASConfig, package="model")
cs.store(group="model", name="joint-ctc-attention-transformer", node=JointCTCAttentionTransformerConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train")
def main(config: DictConfig) -> None:
    warnings.filterwarnings('ignore')
    logger.info(OmegaConf.to_yaml(config))
    train(config)


if __name__ == '__main__':
    main()
