""" Implementation of all available options """
from e2e.modules.definition import logger


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during inference.
    """
    group = parser.add_argument_group('Model')
    group.add_argument('--use_bidirectional', action='store_true', default=False)
    group.add_argument('--hidden_dim', type=int, default=256, help='hidden state dimension of model (default: 256)')
    group.add_argument('--dropout', type=float, default=0.3, help='dropout ratio in training (default: 0.3)')
    group.add_argument('--num_heads', type=int, default=4, help='number of head in attention (default: 4)')
    group.add_argument('--label_smoothing', type=float, default=0.1, help='ratio of label smoothing (default: 0.1)')
    group.add_argument('--listener_layer_size', type=int, default=5, help='layer size of encoder (default: 5)')
    group.add_argument('--speller_layer_size', type=int, default=3, help='layer size of decoder (default: 3)')
    group.add_argument('--rnn_type', type=str, default='gru', help='type of rnn cell: [gru, lstm, rnn] (default: gru)')
    group.add_argument('--teacher_forcing_ratio', type=float, default=0.99,
                       help='teacher forcing ratio in decoder (default: 0.99)')


def train_opts(parser):
    """ Training and saving options """
    group = parser.add_argument_group('General')
    group.add_argument('--use_multi_gpu', action='store_true', default=False)
    group.add_argument('--init_uniform', action='store_true', default=False)
    group.add_argument('--use_augment', action='store_true', default=False)
    group.add_argument('--use_pickle', action='store_true', default=False)
    group.add_argument('--use_cuda', action='store_true', default=False)
    group.add_argument('--augment_num', type=int, default=1, help='Number of SpecAugemnt per data (default: 1)')
    group.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    group.add_argument('--num_workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    group.add_argument('--num_epochs', type=int, default=20, help='number of epochs in training (default: 20)')
    group.add_argument('--lr', type=float, default=3e-04, help='initial learning rate (default: 3e-04)')
    group.add_argument('--min_lr', type=float, default=3e-05, help='minimum learning rate (default: 3e-05)')
    group.add_argument('--lr_factor', type=float, default=1 / 3, help='minimum learning rate (default: 1/3)')
    group.add_argument('--lr_patience', type=int, default=1,
                       help=' Number of epochs with no improvement after which learning rate will be reduced. (default: 1)')
    group.add_argument('--valid_ratio', type=float, default=0.01, help='validation dataset ratio in training dataset')
    group.add_argument('--max_len', type=int, default=151, help='maximum characters of sentence (default: 151)')
    group.add_argument('--seed', type=int, default=7, help='random seed (default: 7)')
    group.add_argument('--save_result_every', type=int, default=1000,
                       help='to determine whether to store training results every N timesteps (default: 1000)')
    group.add_argument('--checkpoint_every', type=int, default=5000,
                       help='to determine whether to store training checkpoint every N timesteps (default: 5000)')
    group.add_argument('--print_every', type=int, default=10,
                       help='to determine whether to store training progress every N timesteps (default: 10')
    group.add_argument('--resume', action='store_true', default=False,
                       help='Indicates if training has to be resumed from the latest checkpoint')


def preprocess_opts(parser):
    """ Pre-processing options """
    group = parser.add_argument_group('Input')
    group.add_argument('--sr', type=int, default=16000, help='sample rate (default: 16000)')
    group.add_argument('--window_size', type=int, default=20, help='Window size for spectrogram (default: 20ms)')
    group.add_argument('--stride', type=int, default=10, help='Window stride for spectrogram (default: 10ms)')
    group.add_argument('--n_mels', type=int, default=80, help='number of mel filter (default: 80)')
    group.add_argument('--normalize', action='store_true', default=False)
    group.add_argument('--del_silence', action='store_true', default=False)
    group.add_argument('--input_reverse', action='store_true', default=False)
    group.add_argument('--feature_extract_by', type=str, default='librosa',
                       help='which library to use for feature extraction: [librosa, torchaudio] (default: librosa)')
    group.add_argument('--time_mask_para', type=int, default=50,
                       help='Hyper Parameter for Time Masking to limit time masking length (default: 50)')
    group.add_argument('--freq_mask_para', type=int, default=12,
                       help='Hyper Parameter for Freq Masking to limit freq masking length (default: 12)')
    group.add_argument('--time_mask_num', type=int, default=2,
                       help='how many time-masked area to make (default: 2)')
    group.add_argument('--freq_mask_num', type=int, default=2,
                       help='how many freq-masked area to make (default: 2)')


def inference_opts(parser):
    """ inference options """
    group = parser.add_argument_group('Infer')
    group.add_argument('--use_multi_gpu', action='store_true', default=False)
    group.add_argument('--num_workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    group.add_argument('--use_cuda', action='store_true', default=False)
    group.add_argument('--model_path', type=str, default=None, help='Location to load models (default: None')
    group.add_argument('--batch_size', type=int, default=1, help='batch size in inference (default: 1)')
    group.add_argument('--k', type=int, default=5, help='size of beam (default: 5)')
    group.add_argument('--use_beam_search', action='store_true', default=True)
    group.add_argument('--print_every', type=int, default=10,
                       help='to determine whether to store inference progress every N timesteps (default: 10')


def print_preprocess_opts(opt):
    """ Print preprocess options """
    logger.info('--mode: %s' % str(opt.mode))
    logger.info('--sr: %s' % str(opt.sr))
    logger.info('--window_size: %s' % str(opt.window_size))
    logger.info('--stride: %s' % str(opt.stride))
    logger.info('--n_mels: %s' % str(opt.n_mels))
    logger.info('--normalize: %s' % str(opt.normalize))
    logger.info('--del_silence: %s' % str(opt.del_silence))
    logger.info('--input_reverse: %s' % str(opt.input_reverse))
    logger.info('--feature_extract_by: %s' % str(opt.feature_extract_by))
    logger.info('--time_mask_para: %s' % str(opt.time_mask_para))
    logger.info('--freq_mask_para: %s' % str(opt.freq_mask_para))
    logger.info('--time_mask_num: %s' % str(opt.time_mask_num))
    logger.info('--freq_mask_num: %s' % str(opt.freq_mask_num))


def print_model_opts(opt):
    """ Print model options """
    logger.info('--use_bidirectional: %s' % str(opt.use_bidirectional))
    logger.info('--hidden_dim: %s' % str(opt.hidden_dim))
    logger.info('--dropout: %s' % str(opt.dropout))
    logger.info('--num_heads: %s' % str(opt.num_heads))
    logger.info('--label_smoothing: %s' % str(opt.label_smoothing))
    logger.info('--listener_layer_size: %s' % str(opt.listener_layer_size))
    logger.info('--speller_layer_size: %s' % str(opt.speller_layer_size))
    logger.info('--rnn_type: %s' % str(opt.rnn_type))
    logger.info('--teacher_forcing_ratio: %s' % str(opt.teacher_forcing_ratio))


def print_train_opts(opt):
    """ Print train options """
    logger.info('--use_multi_gpu: %s' % str(opt.use_multi_gpu))
    logger.info('--init_uniform: %s' % str(opt.init_uniform))
    logger.info('--use_augment: %s' % str(opt.use_augment))
    logger.info('--use_pickle: %s' % str(opt.use_pickle))
    logger.info('--use_cuda: %s' % str(opt.use_cuda))
    logger.info('--augment_num: %s' % str(opt.augment_num))
    logger.info('--batch_size: %s' % str(opt.batch_size))
    logger.info('--num_workers: %s' % str(opt.num_workers))
    logger.info('--num_epochs: %s' % str(opt.num_epochs))
    logger.info('--lr: %s' % str(opt.lr))
    logger.info('--min_lr: %s' % str(opt.min_lr))
    logger.info('--lr_factor: %s' % str(opt.lr_factor))
    logger.info('--lr_patience: %s' % str(opt.lr_patience))
    logger.info('--lr_factor: %s' % str(opt.lr_factor))
    logger.info('--valid_ratio: %s' % str(opt.valid_ratio))
    logger.info('--max_len: %s' % str(opt.max_len))
    logger.info('--seed: %s' % str(opt.seed))
    logger.info('--save_result_every: %s' % str(opt.save_result_every))
    logger.info('--checkpoint_every: %s' % str(opt.checkpoint_every))
    logger.info('--print_every: %s' % str(opt.print_every))
    logger.info('--resume: %s' % str(opt.resume))


def print_inference_opts(opt):
    """ Print inference options """
    logger.info('--use_multi_gpu: %s' % str(opt.use_multi_gpu))
    logger.info('--num_workers: %s' % str(opt.num_workers))
    logger.info('--use_cuda: %s' % str(opt.use_cuda))
    logger.info('--model_path: %s' % str(opt.model_path))
    logger.info('--batch_size: %s' % str(opt.batch_size))
    logger.info('--use_beam_search: %s' % str(opt.use_beam_search))
    logger.info('--k: %s' % str(opt.k))
    logger.info('--print_every: %s' % str(opt.print_every))


def print_opts(opt, mode='train'):
    """ Print options """
    if mode == 'train':
        print_preprocess_opts(opt)
        print_model_opts(opt)
        print_train_opts(opt)

    elif mode == 'eval':
        print_preprocess_opts(opt)
        print_inference_opts(opt)
