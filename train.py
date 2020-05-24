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
import inspect
import random
import torch
import warnings
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from e2e.data_loader.data_loader import split_dataset, load_data_list
from e2e.loss.loss import LabelSmoothingLoss
from e2e.modules.checkpoint import Checkpoint
from e2e.modules.utils import check_envirionment
from e2e.trainer.supervised_trainer import SupervisedTrainer
from e2e.modules.model_builder import build_model
from e2e.modules.opts import print_opts, train_opts, model_opts, preprocess_opts
from e2e.modules.global_var import PAD_token, char2id


def train(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = check_envirionment(opt.use_cuda)

    audio_paths, script_paths = load_data_list(opt.data_list_path, opt.dataset_path)

    epoch_time_step, trainset_list, validset = split_dataset(opt, audio_paths, script_paths)
    model = build_model(opt, device)

    if opt.use_multi_gpu:
        optimizer = optim.Adam(model.module.parameters(), lr=opt.lr)

    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=opt.lr_patience,
        factor=opt.lr_factor,
        verbose=True,
        min_lr=opt.min_lr
    )

    if opt.label_smoothing == 0.0:
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
    else:
        criterion = LabelSmoothingLoss(len(char2id), PAD_token, opt.label_smoothing, dim=-1).to(device)

    trainer = SupervisedTrainer(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        trainset_list=trainset_list,
        validset=validset,
        num_workers=opt.num_workers,
        device=device,
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
    Checkpoint(model, model.op.optimizer, model.lr_scheduler, model.criterion,
               model.trainset_list, model.validset, opt.num_epochs).save()


def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--mode', type=str, default='train')

    preprocess_opts(parser)
    model_opts(parser)
    train_opts(parser)

    return parser


def main():
    warnings.filterwarnings('ignore')
    parser = _get_parser()
    opt = parser.parse_args()
    print_opts(opt, opt.mode)

    train(opt)


if __name__ == '__main__':
    main()
