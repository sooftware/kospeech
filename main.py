"""
    -*- coding: utf-8 -*-

    @source_code{
      title={End-to-end Speech Recognition},
      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
      link={https://github.com/sooftware/End-to-end-Speech-Recognition},
      year={2020}
    }
"""
import random
import torch
import warnings
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from e2e.dataset.data_loader import split_dataset, load_data_list
from e2e.loss.loss import LabelSmoothingLoss
from e2e.modules.definition import *
from e2e.evaluator.evaluator import Evaluator
from e2e.trainer.supervised_trainer import SupervisedTrainer
from e2e.modules.model_builder import build_model, load_test_model
from e2e.modules.opts import get_parser, print_opts


def check_envirionment(opt):
    cuda = opt.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info("device : %s" % torch.cuda.get_device_name(idx))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    else:
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("PyTorch version : %s" % torch.__version__)

    return device


def train(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    device = check_envirionment(opt)

    audio_paths, label_paths = load_data_list(TRAIN_LIST_PATH, DATASET_PATH)
    total_time_step, trainset_list, validset = split_dataset(opt, audio_paths, label_paths)
    model = build_model(opt, device)

    optimizer = optim.Adam(model.module.parameters(), lr=opt.lr)
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
    model = trainer.train(model, opt.batch_size, total_time_step, opt.num_epochs,
                          teacher_forcing_ratio=opt.teacher_forcing_ratio, resume=opt.resume)
    torch.save(model, './data/weight_file/model.pt')


def evaluate(opt):
    device = check_envirionment(opt)

    model = load_test_model(opt, device, use_beamsearch=True)
    evaluator = Evaluator(batch_size=1, device=device)
    evaluator.evaluate(model, opt, TEST_LIST_PATH, DATASET_PATH)


def main():
    warnings.filterwarnings('ignore')
    parser = get_parser()
    opt = parser.parse_args()
    print_opts(opt, opt.mode)

    if opt.mode == 'train':
        train(opt)

    elif opt.mode == 'eval':
        evaluate(opt)

    else:
        raise ValueError("Unsupported mode: {0}".format(opt.mode))


if __name__ == '__main__':
    main()
