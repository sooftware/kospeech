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

import torch
import torch.nn as nn
import logging
import platform
from omegaconf import DictConfig
from kospeech.vocabs import Vocabulary
from torch import optim
from kospeech.optim import (
    RAdam,
    AdamP,
    Novograd,
)
from kospeech.criterion import (
    LabelSmoothedCrossEntropyLoss,
    JointCTCCrossEntropyLoss,
)
import pdb

logger = logging.getLogger(__name__)


def check_envirionment(use_cuda: bool) -> torch.device:
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return device


def get_optimizer(model: nn.Module, config: DictConfig):
    supported_optimizer = {
        'adam': optim.Adam,
        'radam': RAdam,
        'adamp': AdamP,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'novograd': Novograd,
    }
    assert config.train.optimizer.lower() in supported_optimizer.keys(), \
        f"Unsupported Optimizer: {config.train.optimizer}\n" \
        f"Supported Optimizer: {supported_optimizer.keys()}"
    
    return supported_optimizer[config.train.optimizer](
        model.module.parameters(),
        lr=config.train.init_lr,
        weight_decay=config.train.weight_decay,
    )


def get_criterion(config: DictConfig, vocab: Vocabulary) -> nn.Module:
    if config.model.architecture in ('deepspeech2', 'jasper'):
        criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.train.reduction, zero_infinity=True)
    elif config.model.architecture in ('las', 'transformer') and config.model.joint_ctc_attention:
        criterion = JointCTCCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            reduction=config.train.reduction,
            ctc_weight=config.model.ctc_weight,
            cross_entropy_weight=config.model.cross_entropy_weight,
            blank_id=vocab.blank_id,
            dim=-1,
            architecture=config.model.architecture,
            smoothing=config.train.label_smoothing,
        )
    elif config.model.architecture == 'transformer' and config.train.label_smoothing <= 0.0:
        criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_id,
            reduction=config.train.reduction
        )
    else:
        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            smoothing=config.train.label_smoothing,
            reduction=config.train.reduction,
            architecture=config.model.architecture,
            dim=-1,
        )

    return criterion
