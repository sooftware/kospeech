# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import warnings
import torch
import hydra
sys.path.append('..')
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from kospeech.data.data_loader import split_dataset
from kospeech.optim import Optimizer
from kospeech.optim.lr_scheduler import TriStageLRScheduler
from kospeech.trainer import SupervisedTrainer
from kospeech.model_builder import build_model
from kospeech.utils import (
    check_envirionment,
    get_optimizer,
    get_criterion,
)
from kospeech.vocabs import (
    KsponSpeechVocabulary,
    LibriSpeechVocabulary,
)
from bin.dataclass import (
    AudioConfig,
    TrainConfig,
    DeepSpeech2Config,
    JointCTCAttentionConfig,
    ListenAttendSpellConfig,
    TransformerConfig,
    Config,
)


def train(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = check_envirionment(opt.use_cuda)

    if opt.dataset == 'kspon':
        if opt.output_unit == 'subword':
            vocab = KsponSpeechVocabulary(vocab_path='../data/vocab/kspon_sentencepiece.vocab',
                                          output_unit=opt.output_unit,
                                          sp_model_path='../data/vocab/kspon_sentencepiece.model')
        else:
            vocab = KsponSpeechVocabulary(
                f'../data/vocab/aihub_{opt.output_unit}_vocabs.csv', output_unit=opt.output_unit
            )

    elif opt.dataset == 'libri':
        vocab = LibriSpeechVocabulary('../data/vocab/tokenizer.vocab', '../data/vocab/tokenizer.model')

    else:
        raise ValueError("Unsupported Dataset : {0}".format(opt.dataset))

    if not opt.resume:
        epoch_time_step, trainset_list, validset = split_dataset(opt, opt.transcripts_path, vocab)
        model = build_model(opt, vocab, device)

        optimizer = get_optimizer(model, opt)

        lr_scheduler = TriStageLRScheduler(
            optimizer=optimizer,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr,
            final_lr=opt.final_lr,
            init_lr_scale=opt.init_lr_scale,
            final_lr_scale=opt.final_lr_scale,
            warmup_steps=opt.warmup_steps,
            total_steps=int(opt.num_epochs * epoch_time_step)
        )
        optimizer = Optimizer(optimizer, lr_scheduler, opt.warmup_steps, opt.max_grad_norm)
        criterion = get_criterion(opt, vocab)

    else:
        trainset_list = None
        validset = None
        model = None
        optimizer = None
        epoch_time_step = None
        criterion = get_criterion(opt, vocab)

    trainer = SupervisedTrainer(
        optimizer=optimizer,
        criterion=criterion,
        trainset_list=trainset_list,
        validset=validset,
        num_workers=opt.num_workers,
        device=device,
        teacher_forcing_step=opt.teacher_forcing_step,
        min_teacher_forcing_ratio=opt.min_teacher_forcing_ratio,
        print_every=opt.print_every,
        save_result_every=opt.save_result_every,
        checkpoint_every=opt.checkpoint_every,
        architecture=opt.architecture,
        vocab=vocab,
        joint_ctc_attention=opt.joint_ctc_attention
    )
    model = trainer.train(
        model=model,
        batch_size=opt.batch_size,
        epoch_time_step=epoch_time_step,
        num_epochs=opt.num_epochs,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        resume=opt.resume
    )
    return model


cs = ConfigStore.instance()
cs.store(group="audio", name="default", node=AudioConfig, package="audio")
cs.store(group="train", name="default", node=TrainConfig, package="train")
cs.store(group="model", name="ds2", node=DeepSpeech2Config, package="model")
cs.store(group="model", name="joint-ctc-attention", node=JointCTCAttentionConfig, package="model")
cs.store(group="model", name="las", node=ListenAttendSpellConfig, package="model")
cs.store(group="model", name="transformer", node=TransformerConfig, package="model")
cs.store(name="config", node=Config)


@hydra.main(config_path=os.path.join('..', "config"), config_name="default")
def main(config: OmegaConf):
    warnings.filterwarnings('ignore')

    config = OmegaConf.to_yaml(config)
    print(config)

    #train(opt)


if __name__ == '__main__':
    main()
