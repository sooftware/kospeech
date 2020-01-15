"""
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
#-*- coding: utf-8 -*-
import queue
import torch.nn as nn
import torch.optim as optim
import random
import torch
import time
import pandas as pd
from definition import *
from data.split_dataset import split_dataset
from hyperParams import HyperParams
from loader.baseLoader import BaseDataLoader
from loader.loader import load_data_list, load_targets
from loader.multiLoader import MultiLoader
from models.decoderRNN import DecoderRNN
from models.encoderRNN import EncoderRNN
from models.seq2seq import Seq2seq
from train.evaluate import evaluate
from train.training import train
import os

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logger.info("device : ", torch.cuda.get_device_name(0))
    logger.info("CUDA is available : ", str(torch.cuda.is_available()))
    logger.info("CUDA version : ", torch.versopm.cuda)
    logger.info("PyTorch version : " + torch.__version__)

    train_result = {'loss': [], 'cer': []}
    eval_result = {'loss': [], 'cer': []}

    hparams = HyperParams()
    hparams.input_params()
    hparams.print_hparams()

    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    cuda = not hparams.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    feature_size = 33

    enc = EncoderRNN(feature_size, hparams.hidden_size,
                     input_dropout_p = hparams.dropout, dropout_p = hparams.dropout,
                     n_layers = hparams.encoder_layer_size,
                     bidirectional = hparams.bidirectional, rnn_cell = 'gru', variable_lengths = False)

    dec = DecoderRNN(vocab_size=len(char2index), max_len=hparams.max_len,
                     hidden_size=hparams.hidden_size * (2 if hparams.bidirectional else 1),
                     sos_id=SOS_token, eos_id=EOS_token,
                     layer_size = hparams.decoder_layer_size, rnn_cell = 'gru', bidirectional = hparams.bidirectional,
                     input_dropout_p = hparams.dropout, dropout_p = hparams.dropout, use_attention = hparams.attention)

    model = Seq2seq(enc, dec)
    model.flatten_parameters()
    model = nn.DataParallel(model).to(device) # 병렬처리 부분인 듯

    # Optimize Adam Algorithm
    optimizer = optim.Adam(model.module.parameters(), lr = hparams.lr)
    # CrossEntropy로 loss 계산
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    # load audio_paths & label_paths
    if hparams.mode == 'train':
        audio_paths, label_paths = load_data_list(data_list_path=TRAIN_LIST_PATH, dataset_path=DATASET_PATH)
    else:
        audio_paths, label_paths = load_data_list(data_list_path=TEST_LIST_PATH, dataset_path=DATASET_PATH)
    # load all target scripts for reducing disk i/o
    target_dict = load_targets(label_paths)

    # 데이터 로드 end
    train_batch_num, train_dataset_list, valid_dataset = \
        split_dataset(hparams, audio_paths, label_paths, valid_ratio = 0.05, target_dict = target_dict)


    logger.info('start')
    train_begin = time.time()

    for epoch in range(hparams.max_epochs):
        train_queue = queue.Queue(hparams.workers * 2)
        train_loader = MultiLoader(train_dataset_list, train_queue, hparams.batch_size, hparams.workers)
        train_loader.start()
        train_loss, train_cer = train(model, train_batch_num,
                                      train_queue, criterion,
                                      optimizer, device,
                                      train_begin, hparams.workers,
                                      10, hparams.teacher_forcing)

        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_loader.join()

        valid_queue = queue.Queue(hparams.workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, hparams.batch_size, 0)
        valid_loader.start()

        eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        train_result["loss"].append(train_loss)
        train_result["cer"].append(train_cer)
        eval_result["loss"].append(eval_loss)
        eval_result["cer"].append(eval_cer)

        valid_loader.join()

        torch.save(model, "./weight_file/epoch" + str(epoch))
        train_result = pd.DataFrame(train_result)
        eval_result = pd.DataFrame(eval_result)
        train_result.to_csv("./csv/train_result.csv", encoding='cp949', index=False)
        eval_result.to_csv("./csv/eval_result.csv", encoding='cp949', index=False)