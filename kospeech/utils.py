# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
import logging
import platform

from omegaconf import DictConfig
from kospeech.vocabs import Vocabulary
from torch import optim
from kospeech.criterion import (
    LabelSmoothedCrossEntropyLoss,
    JointCTCCrossEntropyLoss
)
from kospeech.optim import (
    RAdam,
    AdamP
)


class Logger(object):
    """
    Print log message in format.
    FORMAT: [%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s
    """
    def __init__(self):
        self.logger = logging.getLogger('root')
        FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
        self.logger.setLevel(logging.INFO)

    def info(self, message=''):
        """ Print log message for information """
        self.logger.info(message)

    def debug(self, message=''):
        """ Print log message for debugging """
        self.logger.debug(message)


logger = Logger()


def check_envirionment(use_cuda: bool):
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info("Operating System : %s %s" % (platform.system(), platform.release()))
    logger.info("Processor : %s" % platform.processor())

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


def get_optimizer(model: nn.Module, config: DictConfig):
    if config.train.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.module.parameters(), lr=config.train.init_lr, weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer.lower() == 'radam':
        optimizer = RAdam(
            model.module.parameters(), lr=config.train.init_lr, weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer.lower() == 'adamp':
        optimizer = AdamP(
            model.module.parameters(), lr=config.train.init_lr, weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.module.parameters(), lr=config.train.init_lr, weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(
            model.module.parameters(), lr=config.train.init_lr, weight_decay=config.train.weight_decay
        )
    else:
        raise ValueError(f"Unsupported Optimizer, Supported Optimizer : Adam, RAdam, Adadelta, Adagrad")

    return optimizer


def get_criterion(config: DictConfig, vocab: Vocabulary):
    if config.model.architecture == 'deepspeech2':
        criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.train.reduction, zero_infinity=True)
    elif config.model.architecture == 'las' and config.model.joint_ctc_attention:
        criterion = JointCTCCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            reduction=config.train.reduction,
            ctc_weight=config.model.ctc_weight,
            cross_entropy_weight=config.model.cross_entropy_weight,
            blank_id=vocab.blank_id,
            dim=-1,
        )
    elif config.model.architecture == 'transformer':
        criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_id,
            reduction=config.train.reduction
        )
    else:
        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            smoothing=config.model.label_smoothing,
            reduction=config.train.reduction,
            architecture=config.model.architecture,
            dim=-1
        )

    return criterion
