"""
-*- coding: utf-8 -*-

@source_code{
  title={End-to-end Speech Recognition},
  author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
  link={https://github.com/sooftware/End-to-end-Speech-Recognition},
  year={2020}
}
"""
import argparse
import random
import torch
import warnings
from torch import optim, nn
from e2e.data.data_loader import split_dataset, load_data_list
from e2e.optim.loss import LabelSmoothingLoss
from e2e.checkpoint.checkpoint import Checkpoint
from e2e.optim.lr_scheduler import RampUpLR
from e2e.optim.optim import Optimizer
from e2e.trainer.supervised_trainer import SupervisedTrainer
from e2e.model_builder import build_model
from e2e.opts import print_opts, build_train_opts, build_model_opts, build_preprocess_opts
from e2e.utils import PAD_token, char2id, check_envirionment


def train(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = check_envirionment(opt.use_cuda)

    audio_paths, script_paths = load_data_list(opt.data_list_path, opt.dataset_path)

    epoch_time_step, trainset_list, validset = split_dataset(opt, audio_paths, script_paths)
    model = build_model(opt, device)

    optimizer = optim.Adam(model.module.parameters(), lr=opt.init_lr)

    if opt.rampup:
        scheduler = RampUpLR(optimizer, opt.init_lr, opt.high_plateau_lr, opt.rampup_period)
        optimizer = Optimizer(optimizer, scheduler, opt.rampup_period, opt.max_grad_norm)
    else:
        optimizer = Optimizer(optimizer, None, 0, opt.max_grad_norm)

    if opt.label_smoothing == 0.0:
        criterion = nn.NLLLoss(reduction='sum', ignore_index=PAD_token).to(device)
    else:
        criterion = LabelSmoothingLoss(len(char2id), PAD_token, opt.label_smoothing, dim=-1).to(device)

    trainer = SupervisedTrainer(
        optimizer=optimizer,
        criterion=criterion,
        trainset_list=trainset_list,
        validset=validset,
        num_workers=opt.num_workers,
        high_plateau_lr=opt.high_plateau_lr,
        low_plateau_lr=opt.low_plateau_lr,
        decay_threshold=opt.decay_threshold,
        exp_decay_period=opt.exp_decay_period,
        device=device,
        teacher_forcing_step=opt.teacher_forcing_step,
        min_teacher_forcing_ratio=opt.min_teacher_forcing_ratio,
        print_every=opt.print_every,
        save_result_every=opt.save_result_every,
        checkpoint_every=opt.checkpoint_every
    )
    model = trainer.train(
        model=model,
        batch_size=opt.batch_size,
        epoch_time_step=epoch_time_step,
        num_epochs=opt.num_epochs,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        resume=opt.resume
    )
    Checkpoint(model, model.optimizer, model.criterion, model.trainset_list, model.validset, opt.num_epochs).save()


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
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
