# -*- coding: utf-8 -*-
"""
- Korean Speech Recognition
Team: Kai.Lib
    ● Team Member
        ○ Kim-Soo-Hwan: KW University elcomm. senior
        ○ Bae-Se-Young: KW University elcomm. senior
        ○ Won-Cheol-Hwang: KW University elcomm. senior

Model Architecture:
    ● Listen, Attend and Spell (Seq2seq with Attention)

Data:
    ● A.I Hub Dataset

Score:
    ● CRR: Character Recognition Rate
    ● CER: Character Error Rate based on Edit Distance

GitHub repository : https://github.com/sh951011/Korean-ASR
Documentation : https://sh951011.github.io/Korean-Speech-Recognition/index.html

Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import queue
import torch.nn as nn
import torch.optim as optim
import random
import torch
import time
import os
from models.speller import Speller
from models.listener import Listener
from models.listenAttendSpell import ListenAttendSpell
from package.checkpoint import CheckPoint
from package.dataset import split_dataset
from package.definition import *
from package.evaluator import evaluate
from package.hparams import HyperParams
from package.loader import load_data_list, load_targets, load_pickle, MultiLoader, BaseDataLoader
from package.loss import LabelSmoothingLoss
from package.trainer import supervised_train
from package.utils import save_epoch_result

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # if you use Multi-GPU, delete this line
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % (torch.version.cuda))
    logger.info("PyTorch version : %s" % (torch.__version__))

    checkpoint = CheckPoint()
    snapshot = checkpoint.load("./data/checkpoints/???.bin")

    hparams = snapshot['hparams']
    model = snapshot['model']
    optimizer = snapshot['optimizer']
    criterion = snapshot['criterion']
    train_dataset_list = snapshot['train_dataset_list']
    total_time_step = snapshot['total_time_step']
    valid_dataset = snapshot['valid_dataset']

    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    cuda = hparams.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info('start')
    train_begin = time.time()

    for epoch in range(hparams.max_epochs):
        if epoch != 0:
            train_queue = queue.Queue(hparams.worker_num << 1)
            for train_dataset in train_dataset_list:
                train_dataset.shuffle()
            train_loader = MultiLoader(train_dataset_list, train_queue, hparams.batch_size, hparams.worker_num)
            train_loader.start()
            checkpoint.snapshot['train_dataset_list'] = train_dataset_list
        else:
            train_queue = snapshot['train_queue']

        train_loss, train_cer = supervised_train(
            model = model,
            total_time_step = total_time_step,
            hparams = hparams,
            queue = train_queue,
            criterion = criterion,
            epoch = epoch,
            optimizer = optimizer,
            checkpoint = checkpoint,
            device = device,
            train_begin = train_begin,
            worker_num = hparams.worker_num,
            print_time_step = 10,
            teacher_forcing_ratio = hparams.teacher_forcing,
            snapshot = snapshot
        )
        torch.save(model, "model.pt")
        torch.save(model, "./data/weight_file/epoch%s.pt" % str(epoch))
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
        train_loader.join()
        valid_queue = queue.Queue(hparams.worker_num << 1)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, hparams.batch_size, 0)
        valid_loader.start()

        valid_loss, valid_cer = evaluate(model, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))

        valid_loader.join()

        save_epoch_result(train_result=[train_dict, train_loss, train_cer], valid_result=[valid_dict, valid_loss, valid_cer])
        logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)