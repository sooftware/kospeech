# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
import random
import warnings
import torch
import torch.nn as nn
from torch import optim
sys.path.append('..')
from kospeech.data.data_loader import split_dataset
from kospeech.optim.lr_scheduler import TriStageLRScheduler
from kospeech.trainer import SupervisedTrainer
from kospeech.utils import check_envirionment
from kospeech.model_builder import build_model
from kospeech.criterion import (
    LabelSmoothedCrossEntropyLoss,
    JointCTCAttentionLoss
)
from kospeech.vocabs import (
    KsponSpeechVocabulary,
    LibriSpeechVocabulary
)
from kospeech.optim import (
    Optimizer,
    RAdam,
    AdamP
)
from kospeech.opts import (
    print_opts,
    build_train_opts,
    build_model_opts,
    build_preprocess_opts
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

        if opt.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.module.parameters(), lr=opt.init_lr, weight_decay=opt.weight_decay)
        elif opt.optimizer.lower() == 'radam':
            optimizer = RAdam(model.module.parameters(), lr=opt.init_lr, weight_decay=opt.weight_decay)
        elif opt.optimizer.lower() == 'adamp':
            optimizer = AdamP(model.module.parameters(), lr=opt.init_lr, weight_decay=opt.weight_decay)
        elif opt.optimizer.lower() == 'adadelta':
            optimizer = optim.Adadelta(model.module.parameters(), lr=opt.init_lr, weight_decay=opt.weight_decay)
        elif opt.optimizer.lower() == 'adagrad':
            optimizer = optim.Adagrad(model.module.parameters(), lr=opt.init_lr, weight_decay=opt.weight_decay)
        else:
            raise ValueError(f"Unsupported Optimizer, Supported Optimizer : Adam, RAdam, Adadelta, Adagrad")

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

        if opt.architecture == 'deepspeech2':
            criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=opt.reduction).to(device)
        elif opt.architecture == 'las' and opt.joint_ctc_attention:
            criterion = JointCTCAttentionLoss(
                num_classes=len(vocab),
                ignore_index=vocab.pad_id,
                reduction=opt.reduction,
                ctc_weight=opt.ctc_weight,
                cross_entropy_weight=opt.cross_entropy_weight,
                blank_id=vocab.blank_id,
                dim=-1,
            ).to(device)
        else:
            criterion = LabelSmoothedCrossEntropyLoss(
                num_classes=len(vocab),
                ignore_index=vocab.pad_id,
                smoothing=opt.label_smoothing,
                reduction=opt.reduction,
                architecture=opt.architecture,
                dim=-1
            ).to(device)

    else:
        trainset_list = None
        validset = None
        model = None
        optimizer = None
        epoch_time_step = None

        if opt.architecture == 'deepspeech2':
            criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=opt.reduction).to(device)
        elif opt.architecture == 'las' and opt.joint_ctc_attention:
            criterion = JointCTCAttentionLoss(
                num_classes=len(vocab),
                ignore_index=vocab.pad_id,
                reduction=opt.reduction,
                ctc_weight=opt.ctc_weight,
                cross_entropy_weight=opt.cross_entropy_weight,
                dim=-1,
            ).to(device)
        else:
            criterion = LabelSmoothedCrossEntropyLoss(
                num_classes=len(vocab),
                ignore_index=vocab.pad_id,
                smoothing=opt.label_smoothing,
                reduction=opt.reduction,
                architecture=opt.architecture,
                dim=-1
            ).to(device)

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


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='KoSpeech')
    parser.add_argument('--mode', type=str, default='train')

    build_preprocess_opts(parser)
    build_model_opts(parser)
    build_train_opts(parser)

    return parser


def main():
    warnings.filterwarnings('ignore')
    parser = _get_parser()
    opt = parser.parse_args()
    print_opts(opt, opt.mode)
    train(opt)


if __name__ == '__main__':
    main()
