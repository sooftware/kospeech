"""
-*- coding: utf-8 -*-
- Korean Speech Recognition
Team: Kai.Lib (KwangWoon A.I Library)
    ● Team Member
        ○ Kim-Soo-Hwan: KW University elcomm. senior
        ○ Bae-Se-Young: KW University elcomm. senior
        ○ Won-Cheol-Hwang: KW University elcomm. senior
Model Architecture:
    ● seq2seq with Attention (Listen Attend and Spell)
Data:
    ● Using A.I Hub Dataset
Score:
    ● CRR: Character Recognition Rate
    ● CER: Character Error Rate based on Edit Distance
Reference:
    ● Model
        ○ IBM PyTorch-seq2seq : https://github.com/IBM/pytorch-seq2seq
    ● Dataset
        ○ A.I Hub Korean Voice Dataset : http://www.aihub.or.kr/aidata/105

gitHub repository : https://github.com/sh951011/Korean-ASR

License:
    Copyright 2020- Kai.Lib
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
          http://www.apache.org/licenses/LICENSE-2.0
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
from definition import *
from data.split_dataset import split_dataset
from hyperParams import HyperParams
from loader.baseLoader import BaseDataLoader
from loader.loader import load_data_list, load_targets
from loader.multiLoader import MultiLoader
from models.speller import Speller
from models.listener import Listener
from models.listenAttendSpell import ListenAttendSpell
from train.evaluate import evaluate
from train.save_and_load import save_epoch_result, load_model, load_pickle, save_pickle
from train.training import train

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("device : %s" % torch.cuda.get_device_name(0))
    logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
    logger.info("CUDA version : %s" % (torch.version.cuda))
    logger.info("PyTorch version : %s" % (torch.__version__))

    hparams = HyperParams()
    hparams.logger_hparams()

    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    cuda = hparams.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    feat_size = 33

    listener = Listener(feat_size=feat_size, hidden_size=hparams.hidden_size,
                        dropout_p=hparams.dropout, layer_size=hparams.listener_layer_size,
                        bidirectional=hparams.use_bidirectional, rnn_cell='gru', use_pyramidal=hparams.use_pyramidal)

    speller = Speller(vocab_size=len(char2index), max_len=hparams.max_len,
                      hidden_size=hparams.hidden_size * (2 if hparams.use_bidirectional else 1),
                      sos_id=SOS_token, eos_id=EOS_token, layer_size = hparams.speller_layer_size,
                      rnn_cell = 'gru', dropout_p = hparams.dropout, use_attention = hparams.use_attention)

    if hparams.load_model:
        model = load_model(hparams.model_path)
    else:
        model = ListenAttendSpell(listener=listener, speller=speller)
        model.flatten_parameters()
        model = nn.DataParallel(model).to(device)

    # Optimize Adam Algorithm
    optimizer = optim.Adam(model.module.parameters(), lr=hparams.lr)
    # CrossEntropy로 loss 계산
    loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    # load audio_paths & label_paths
    audio_paths, label_paths = load_data_list(data_list_path=TRAIN_LIST_PATH, dataset_path=DATASET_PATH)

    if hparams.use_pickle:
        target_dict = load_pickle("./pickle/target_dict.txt", "load all target_dict using pickle complete !!")
    else:
        logger.info("load all target dictionary for reducing disk I/O")
        target_dict = load_targets(label_paths)
        save_pickle(target_dict, "./pickle/target_dict.txt", "dump all target dictionary using pickle complete !!")

    logger.info("split dataset start !!")
    train_batch_num, train_dataset_list, valid_dataset = \
        split_dataset(hparams, audio_paths, label_paths, valid_ratio=0.05, target_dict=target_dict)
    logger.info("split dataset complete !!")

    logger.info('start')
    train_begin = time.time()

    for epoch in range(hparams.max_epochs):
        train_queue = queue.Queue(hparams.worker_num * 2)
        for train_dataset in train_dataset_list:
            train_dataset.shuffle()
        train_loader = MultiLoader(train_dataset_list, train_queue, hparams.batch_size, hparams.worker_num)
        train_loader.start()
        train_loss, train_cer = train(model=model, total_batch_size=train_batch_num,
                                      queue=train_queue, loss_func=loss_func,
                                      optimizer=optimizer, device=device,
                                      train_begin=train_begin, worker_num=hparams.worker_num,
                                      print_batch=10, teacher_forcing_ratio=hparams.teacher_forcing)
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
        train_loader.join()
        valid_queue = queue.Queue(hparams.worker_num * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, hparams.batch_size, 0)
        valid_loader.start()

        valid_loss, valid_cer = evaluate(model, valid_queue, loss_func, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))

        valid_loader.join()
        torch.save(model, "./weight_file/epoch%s" % str(epoch))

        save_epoch_result(train_result=[train_dict, train_loss, train_cer], valid_result=[valid_dict, valid_loss, valid_cer])
        logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)