"""
    -*- coding: utf-8 -*-

    @source_code{
      title={Character-unit based End-to-end Korean Speech Recognition},
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
import argparse
import warnings
from definition import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.speller import Speller
from model.listener import Listener
from model.listenAttendSpell import ListenAttendSpell
from trainer import supervised_train, evaluate
from loss import LabelSmoothingLoss
from utils import save_epoch_result
from label_loader import load_targets
from data_loader import split_dataset, load_data_list, load_pickle, MultiLoader, AudioDataLoader


parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
parser.add_argument('--use_bidirectional', action='store_true', default=True)
parser.add_argument('--input_reverse', action='store_true', default=False)
parser.add_argument('--use_augment', action='store_true', default=False)
parser.add_argument('--use_pickle', action='store_true', default=False)
parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--run_by_sample', action='store_true', default=True)
parser.add_argument('--model_path', type=str, default=None, help='Location to load models (default: None')
parser.add_argument('--augment_num', type=int, default=1, help='Number of SpecAugemnt per data (default: 1)')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden state dimension of model (default: 256)')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio in training (default: 0.3)')
parser.add_argument('--num_head', type=int, default=4, help='number of head in attention (default: 4)')
parser.add_argument('--attn_dim', type=int, default=128, help='dimention of attention (default: 128)')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='ratio of label smoothing (default: 0.1)')
parser.add_argument('--conv_type', type=str, default='custom',
                    help='conv type of listener [custom, deepspeech2] (default: custom')
parser.add_argument('--listener_layer_size', type=int, default=2, help='layer size of encoder (default: 5)')
parser.add_argument('--speller_layer_size', type=int, default=1, help='layer size of decoder (default: 3)')
parser.add_argument('--rnn_type', type=str, default='gru', help='type of rnn cell: [gru, lstm, rnn] (default: gru)')
parser.add_argument('--k', type=int, default=5, help='size of beam (default: 5)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size in training (default: 32)')
parser.add_argument('--worker_num', type=int, default=4, help='number of workers in dataset loader (default: 4)')
parser.add_argument('--max_epochs', type=int, default=20, help='number of max epochs in training (default: 20)')
parser.add_argument('--lr', type=float, default=3e-04, help='learning rate (default: 3e-04)')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.99,
                    help='teacher forcing ratio in decoder (default: 0.99)')
parser.add_argument('--valid_ratio', type=float, default=0.01, help='validation dataset ratio in training dataset')
parser.add_argument('--max_len', type=int, default=151, help='maximum characters of sentence (default: 151)')
parser.add_argument('--seed', type=int, default=7, help='random seed (default: 7)')
parser.add_argument('--sr', type=int, default=16000, help='sample rate (default: 16000)')
parser.add_argument('--window_size', type=int, default=20, help='Window size for spectrogram (default: 20ms)')
parser.add_argument('--stride', type=int, default=10, help='Window stride for spectrogram (default: 10ms)')
parser.add_argument('--n_mels', type=int, default=80, help='number of mel filter (default: 80)')
parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--del_silence', action='store_true', default=False)
parser.add_argument('--feature_extract_by', type=str, default='librosa',
                    help='which library to use for feature extraction: [librosa, torchaudio] (default: librosa)')
parser.add_argument('--time_mask_para', type=int, default=50,
                    help='Hyper Parameter for Time Masking to limit time masking length (default: 50)')
parser.add_argument('--freq_mask_para', type=int, default=12,
                    help='Hyper Parameter for Freq Masking to limit freq masking length (default: 12)')
parser.add_argument('--time_mask_num', type=int, default=2,
                    help='how many time-masked area to make (default: 2)')
parser.add_argument('--freq_mask_num', type=int, default=2,
                    help='how many freq-masked area to make (default: 2)')
parser.add_argument('--save_result_every', type=int, default=1000,
                    help='to determine whether to store training results every N timesteps (default: 1000)')
parser.add_argument('--save_model_every', type=int, default=10000,
                    help='to determine whether to store training model every N timesteps (default: 10000)')
parser.add_argument('--print_every', type=int, default=10,
                    help='to determine whether to store training progress every N timesteps (default: 10')


def main():
    warnings.filterwarnings('ignore')
    args = parser.parse_args()
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

    if args.load_model:
        model = torch.load(args.model_path).to(device)
    else:
        listener = Listener(
            in_features=args.n_mels,
            hidden_dim=args.hidden_dim,
            dropout_p=args.dropout,
            num_layers=args.listener_layer_size,
            bidirectional=args.use_bidirectional,
            conv_type=args.conv_type,
            rnn_type=args.rnn_type,
            device=device
        )
        speller = Speller(
            num_class=len(char2id),
            max_length=args.max_len,
            k=args.k,
            hidden_dim=args.hidden_dim << (1 if args.use_bidirectional else 0),
            sos_id=SOS_token,
            eos_id=EOS_token,
            num_layers=args.speller_layer_size,
            rnn_type=args.rnn_type,
            dropout_p=args.dropout,
            num_head=args.num_head,
            attn_dim=args.attn_dim,
            device=device,
            ignore_index=char2id[' ']
        )
        model = ListenAttendSpell(listener, speller)
        model.flatten_parameters()
        model = nn.DataParallel(model).to(device)

        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.333, verbose=True, min_lr=3e-05)

    if args.label_smoothing == 0.0:
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
    else:
        criterion = LabelSmoothingLoss(len(char2id), PAD_token, args.label_smoothing, dim=-1).to(device)

    if args.run_by_sample:
        audio_paths, label_paths = load_data_list(data_list_path=SAMPLE_LIST_PATH, dataset_path=SAMPLE_DATASET_PATH)
    else:
        audio_paths, label_paths = load_data_list(data_list_path=TRAIN_LIST_PATH, dataset_path=DATASET_PATH)

    if args.use_pickle:
        target_dict = load_pickle(TARGET_DICT_PATH, "load all target_dict using pickle complete !!")
    else:
        target_dict = load_targets(label_paths)

    total_time_step, trainset_list, validset = split_dataset(args, audio_paths, label_paths, target_dict)

    logger.info('start')
    train_begin = time.time()

    for epoch in range(args.max_epochs):
        train_queue = queue.Queue(args.worker_num << 1)
        for trainset in trainset_list:
            trainset.shuffle()

        # Training
        train_loader = MultiLoader(trainset_list, train_queue, args.batch_size, args.worker_num)
        train_loader.start()
        train_loss, train_cer = supervised_train(
            model=model,
            total_time_step=total_time_step,
            args=args,
            queue=train_queue,
            criterion=criterion,
            epoch=epoch,
            optimizer=optimizer,
            device=device,
            train_begin=train_begin,
        )
        train_loader.join()

        torch.save(model, "./data/weight_file/epoch%s.pt" % str(epoch))
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        # Validation
        valid_queue = queue.Queue(args.worker_num << 1)
        valid_loader = AudioDataLoader(validset, valid_queue, args.batch_size, 0)
        valid_loader.start()

        valid_loss, valid_cer = evaluate(model, valid_queue, criterion, device)
        valid_loader.join()

        lr_scheduler.step(valid_loss)

        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))
        save_epoch_result(train_result=[train_dict, train_loss, train_cer],
                          valid_result=[valid_dict, valid_loss, valid_cer])
        logger.info('Epoch %d Training result saved as a csv file complete !!' % epoch)


if __name__ == '__main__':
    main()
