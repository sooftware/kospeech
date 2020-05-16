"""
    -*- coding: utf-8 -*-

    @source_code{
      title={Character-unit based End-to-end Korean Speech Recognition},
      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
      link={https://github.com/sooftware/End-to-End-Korean-Speech-Recognition},
      year={2020}
    }
"""

import random
import torch
import warnings
from definition import *
from evaluator import Evaluator
from supervised_trainer import SupervisedTrainer
from utils import print_args
from builder import build_model, load_test_model, build_args


def main():
    warnings.filterwarnings('ignore')
    args = build_args()
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
        model = build_model(args, device)

        trainer = SupervisedTrainer(model, args, device)
        trainer.train(TRAIN_LIST_PATH, DATASET_PATH, 0)

    elif args.mode == 'eval':
        model = load_test_model(args, device, use_beamsearch=True)

        evaluator = Evaluator(model, batch_size=1, device=device)
        evaluator.evaluate(args, TEST_LIST_PATH, DATASET_PATH)

    else:
        raise ValueError("mode should be one of [train, eval]")


if __name__ == '__main__':
    main()
