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
from e2e.modules.utils import print_args
from e2e.modules.builder import build_model, load_test_model, get_parser


def main():
    warnings.filterwarnings('ignore')
    args = get_parser()
    print_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info("device : %s" % torch.cuda.get_device_name(idx))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    if args.mode == 'train':
        audio_paths, label_paths = load_data_list(TRAIN_LIST_PATH, DATASET_PATH)
        total_time_step, trainset_list, validset = split_dataset(args, audio_paths, label_paths)
        model = build_model(args, device)

        optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience,
                                         factor=args.lr_factor, verbose=True, min_lr=args.min_lr)

        if args.label_smoothing == 0.0:
            criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
        else:
            criterion = LabelSmoothingLoss(len(char2id), PAD_token, args.label_smoothing, dim=-1).to(device)

        trainer = SupervisedTrainer(optimizer, lr_scheduler, criterion, trainset_list, validset,
                                    args.num_workers, device,
                                    args.print_every, args.save_result_every, args.checkpoint_every)
        model = trainer.train(model, args.batch_size, total_time_step, args.num_epochs,
                              teacher_forcing_ratio=args.teacher_forcing_ratio, resume=args.resume)
        torch.save(model, './data/weight_file/model.pt')

    elif args.mode == 'eval':
        model = load_test_model(args, device, use_beamsearch=True)
        evaluator = Evaluator(batch_size=1, device=device)
        evaluator.evaluate(model, args, TEST_LIST_PATH, DATASET_PATH)

    else:
        raise ValueError("mode should be one of [train, eval]")


if __name__ == '__main__':
    main()
