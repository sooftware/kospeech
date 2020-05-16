import argparse
import torch
import torch.nn as nn
from las.modules.definition import char2id, EOS_token, SOS_token
from las.model.las import ListenAttendSpell
from las.model.listener import Listener
from las.model.speller import Speller
from las.model.topk_decoder import TopKDecoder

supported_rnns = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN
}

supported_convs = [
    'increase',
    'repeat'
]


def build_model(args, device):
    """ build base model """
    if args.load_model:
        model = torch.load(args.model_path).to(device)

    else:
        listener = build_listener(
            input_size=args.n_mels,
            hidden_dim=args.hidden_dim,
            dropout_p=args.dropout,
            num_layers=args.listener_layer_size,
            bidirectional=args.use_bidirectional,
            rnn_type=args.rnn_type,
            device=device,
            conv_type=args.conv_type
        )
        speller = build_speller(
            num_classes=len(char2id),
            max_length=args.max_len,
            hidden_dim=args.hidden_dim << (1 if args.use_bidirectional else 0),
            sos_id=SOS_token,
            eos_id=EOS_token,
            num_layers=args.speller_layer_size,
            rnn_type=args.rnn_type,
            dropout_p=args.dropout,
            num_heads=args.num_heads,
            device=device
        )
        model = build_las(listener, speller, device, use_multi_gpu=args.use_multi_gpu, init_uniform=args.init_uniform)

    return model


def build_las(listener, speller, device, use_multi_gpu=True, init_uniform=True):
    """ build las model & validate parameters """
    model = ListenAttendSpell(listener, speller)
    model.flatten_parameters()

    if use_multi_gpu:
        model = nn.DataParallel(model).to(device)

    if init_uniform:
        for param in model.parameters():
            param.data.uniform_(-0.08, 0.08)

    return model


def build_listener(input_size, hidden_dim, dropout_p, num_layers, bidirectional, rnn_type, device, conv_type):
    """ build listener & validate parameters """
    assert dropout_p >= 0.0, "dropout probability should be positive"
    assert isinstance(input_size, int), "input_size should be inteager type"
    assert isinstance(num_layers, int), "num_layers should be inteager type"
    assert input_size > 0, "input_size should be greater than 0"
    assert hidden_dim > 0, "hidden_dim should be greater than 0"
    assert rnn_type.lower() in supported_rnns.keys(), "Unsupported RNN Cell: {0}".format(rnn_type)
    assert conv_type.lower() in supported_convs, "Unsupported Conv: {0}".format(conv_type)

    listener = Listener(
        input_size=input_size,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        num_layers=num_layers,
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        device=device,
        conv_type=conv_type
    )

    return listener


def build_speller(num_classes, max_length, hidden_dim, sos_id, eos_id, num_layers, rnn_type, dropout_p, num_heads, device):
    """ build speller & validate parameters """
    assert isinstance(num_classes, int), "num_classes should be inteager type"
    assert isinstance(num_layers, int), "num_layers should be inteager type"
    assert isinstance(sos_id, int), "sos_id should be inteager type"
    assert isinstance(eos_id, int), "eos_id should be inteager type"
    assert isinstance(num_heads, int), "num_heads should be inteager type"
    assert isinstance(max_length, int), "max_length should be inteager type"
    assert isinstance(dropout_p, float), "dropout_p should be inteager type"
    assert hidden_dim % num_heads == 0, "{0} % {1} should be zero".format(hidden_dim, num_heads)
    assert dropout_p >= 0.0, "dropout probability should be positive"
    assert hidden_dim > 0, "hidden_dim should be greater than 0"
    assert rnn_type.lower() in supported_rnns.keys(), "Unsupported RNN Cell: {0}".format(rnn_type)

    speller = Speller(
        num_classes=num_classes,
        max_length=max_length,
        hidden_dim=hidden_dim,
        sos_id=sos_id,
        eos_id=eos_id,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout_p=dropout_p,
        num_heads=num_heads,
        device=device,
    )

    return speller


def load_test_model(args, device, use_beamsearch=True):
    """ load model for performance test """
    model = torch.load(args.model_path)
    model.module.speller.device = device
    model.module.listener.device = device

    if use_beamsearch:
        topk_decoder = TopKDecoder(model.module.speller, args.k)
        model.module.set_speller(topk_decoder)

    return model


def build_args():
    """ build argements """
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--init_uniform', action='store_true', default=False)
    parser.add_argument('--use_bidirectional', action='store_true', default=False)
    parser.add_argument('--input_reverse', action='store_true', default=False)
    parser.add_argument('--use_augment', action='store_true', default=False)
    parser.add_argument('--use_pickle', action='store_true', default=False)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default=None, help='Location to load models (default: None')
    parser.add_argument('--augment_num', type=int, default=1, help='Number of SpecAugemnt per data (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden state dimension of model (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio in training (default: 0.3)')
    parser.add_argument('--num_heads', type=int, default=4, help='number of head in attention (default: 4)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='ratio of label smoothing (default: 0.1)')
    parser.add_argument('--conv_type', type=str, default='increase',
                        help='type of conv in listener [increase, repeat] (default: increase')
    parser.add_argument('--listener_layer_size', type=int, default=5, help='layer size of encoder (default: 5)')
    parser.add_argument('--speller_layer_size', type=int, default=3, help='layer size of decoder (default: 3)')
    parser.add_argument('--rnn_type', type=str, default='gru', help='type of rnn cell: [gru, lstm, rnn] (default: gru)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs in training (default: 20)')
    parser.add_argument('--lr', type=float, default=3e-04, help='initial learning rate (default: 3e-04)')
    parser.add_argument('--min_lr', type=float, default=3e-05, help='minimum learning rate (default: 3e-05)')
    parser.add_argument('--lr_factor', type=float, default=1 / 3, help='minimum learning rate (default: 1/3)')
    parser.add_argument('--lr_patience', type=int, default=1,
                        help=' Number of epochs with no improvement after which learning rate will be reduced. (default: 1)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.99,
                        help='teacher forcing ratio in decoder (default: 0.99)')
    parser.add_argument('--k', type=int, default=5, help='size of beam (default: 5)')
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
    args = parser.parse_args()

    return args
