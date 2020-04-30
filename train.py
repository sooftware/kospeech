"""
    -*- coding: utf-8 -*-

    @source_code{
      title={Character-unit based End-to-End Korean Speech Recognition},
      author={Soohwan Kim, Seyoung Bae, Cheolhwang Won},
      link={https://github.com/sooftware/End-to-End-Korean-Speech-Recognition},
      year={2020}
    }
"""

import queue
import torch.nn as nn
import torch.optim as optim
import random
import torch
import time
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.speller import Speller
from model.listener import Listener
from model.listenAttendSpell import ListenAttendSpell
from utils.dataset import split_dataset
from utils.definition import *
from utils.evaluator import evaluate
from utils.config import Config
from utils.loader import load_data_list, load_targets, load_pickle, MultiLoader, AudioDataLoader
from utils.loss import LabelSmoothingLoss
from utils.trainer import supervised_train
from utils.util import save_epoch_result


if __name__ == '__main__':
    config = Config(
        use_bidirectional=True,
        use_label_smooth=True,
        input_reverse=True,
        use_augment=True,
        use_pickle=True,
        use_cuda=True,
        augment_ratio=1.0,
        hidden_dim=256,
        dropout=0.3,
        n_head=12,
        listener_layer_size=5,
        speller_layer_size=3,
        batch_size=8,
        worker_num=1,
        max_epochs=40,
        use_multistep_lr=False,
        init_lr=0.001,
        high_plateau_lr=0.0003,
        low_plateau_lr=0.00001,
        teacher_forcing=0.99,
        seed=1,
        max_len=151,
        load_model=False,
        model_path=None
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    if device == 'cuda':
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # if you use Multi-GPU, delete this line
        logger.info("device : %s" % torch.cuda.get_device_name(0))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)

    if config.load_model:
        model = torch.load(config.model_path).to(device)
    else:
        listener = Listener(
            in_features=80,
            hidden_dim=config.hidden_dim,
            dropout_p=config.dropout,
            n_layers=config.listener_layer_size,
            bidirectional=config.use_bidirectional,
            rnn_type='gru',
            device=device
        )
        speller = Speller(
            n_class=len(char2id),
            max_length=config.max_len,
            k=5,
            hidden_dim=config.hidden_dim << (1 if config.use_bidirectional else 0),
            sos_id=SOS_token,
            eos_id=EOS_token,
            n_layers=config.speller_layer_size,
            rnn_type='gru',
            dropout_p=config.dropout,
            n_head=config.n_head,
            device=device
        )
        model = ListenAttendSpell(listener, speller)
        model.flatten_parameters()
        model = nn.DataParallel(model).to(device)

        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    optimizer = optim.Adam(model.module.parameters(), lr=config.init_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)

    if config.use_label_smooth:
        criterion = LabelSmoothingLoss(len(char2id), ignore_index=PAD_token, smoothing=0.1, dim=-1).to(device)
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    audio_paths, label_paths = load_data_list(data_list_path=TRAIN_LIST_PATH, dataset_path=DATASET_PATH)

    if config.use_pickle:
        target_dict = load_pickle(TARGET_DICT_PATH, "load all target_dict using pickle complete !!")
    else:
        target_dict = load_targets(label_paths)

    total_time_step, trainset_list, validset = split_dataset(
        config=config,
        audio_paths=audio_paths,
        label_paths=label_paths,
        valid_ratio=0.01,
        target_dict=target_dict,
    )

    logger.info('start')
    train_begin = time.time()

    for epoch in range(config.max_epochs):
        train_queue = queue.Queue(config.worker_num << 1)
        for trainset in trainset_list:
            trainset.shuffle()

        # Training
        train_loader = MultiLoader(trainset_list, train_queue, config.batch_size, config.worker_num)
        train_loader.start()
        train_loss, train_cer = supervised_train(
            model=model,
            total_time_step=total_time_step,
            config=config,
            queue=train_queue,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer,
            device=device,
            train_begin=train_begin,
            worker_num=config.worker_num,
            print_every=10,
            teacher_forcing_ratio=config.teacher_forcing
        )
        train_loader.join()

        torch.save(model, "./data/weight_file/epoch%s.pt" % str(epoch))
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        # Validation
        valid_queue = queue.Queue(config.worker_num << 1)
        valid_loader = AudioDataLoader(validset, valid_queue, config.batch_size, 0)
        valid_loader.start()

        valid_loss, valid_cer = evaluate(model, valid_queue, criterion, device)
        valid_loader.join()

        scheduler.step(valid_loss)

        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))
        save_epoch_result(train_result=[train_dict, train_loss, train_cer],
                          valid_result=[valid_dict, valid_loss, valid_cer])
        logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)
