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
from torch import optim
sys.path.append('..')
from kospeech.data.data_loader import split_dataset, load_data_list
from kospeech.criterion.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss
from kospeech.optim.lr_scheduler import TriStageLRScheduler
from kospeech.optim.optimizer import Optimizer
from kospeech.trainer.supervised_trainer import SupervisedTrainer
from kospeech.model_builder import build_model
from kospeech.opts import (
    print_opts,
    build_train_opts,
    build_model_opts,
    build_preprocess_opts
)
from kospeech.utils import PAD_token, char2id, check_envirionment


def train(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = check_envirionment(opt.use_cuda)

    if not opt.resume:
        audio_paths, script_paths = load_data_list(opt.data_list_path, opt.dataset_path)
        epoch_time_step, trainset_list, validset = split_dataset(opt, audio_paths, script_paths)
        model = build_model(opt, device)

        optimizer = optim.Adam(model.module.parameters(), lr=opt.init_lr, weight_decay=opt.weight_decay)

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

        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(char2id), ignore_index=PAD_token,
            smoothing=opt.label_smoothing, dim=-1,
            reduction=opt.reduction, architecture=opt.architecture
        ).to(device)

    else:
        trainset_list = None
        validset = None
        model = None
        optimizer = None
        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(char2id), ignore_index=PAD_token,
            smoothing=opt.label_smoothing, dim=-1,
            reduction=opt.reduction, architecture=opt.architecture
        ).to(device)
        epoch_time_step = None

    trainer = SupervisedTrainer(
        optimizer=optimizer, criterion=criterion, trainset_list=trainset_list,
        validset=validset, num_workers=opt.num_workers,
        device=device, teacher_forcing_step=opt.teacher_forcing_step,
        min_teacher_forcing_ratio=opt.min_teacher_forcing_ratio, print_every=opt.print_every,
        save_result_every=opt.save_result_every, checkpoint_every=opt.checkpoint_every,
        architecture=opt.architecture
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
