from kospeech.utils import logger


def build_model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during inference.
    """
    group = parser.add_argument_group('Model')
    group.add_argument('--use_bidirectional', '-use_bidirectional',
                       action='store_true', default=False,
                       help='if True, becomes a bidirectional encoder (defulat: False)')
    group.add_argument('--hidden_dim', '-hidden_dim',
                       type=int, default=256,
                       help='hidden state dimension of model (default: 256)')
    group.add_argument('--dropout', '-dropout',
                       type=float, default=0.3,
                       help='dropout ratio in training (default: 0.3)')
    group.add_argument('--num_heads', '-num_heads',
                       type=int, default=8,
                       help='number of head in attention (default: 8)')
    group.add_argument('--label_smoothing', '-label_smoothing',
                       type=float, default=0.1,
                       help='ratio of label smoothing (default: 0.1)')
    group.add_argument('--listener_layer_size', '-listener_layer_size',
                       type=int, default=5,
                       help='layer size of encoder (default: 5)')
    group.add_argument('--speller_layer_size', '-speller_layer_size',
                       type=int, default=3,
                       help='layer size of decoder (default: 3)')
    group.add_argument('--rnn_type', '-rnn_type',
                       type=str, default='gru',
                       help='type of rnn cell: [gru, lstm, rnn] (default: gru)')
    group.add_argument('--extractor', '-extractor',
                       type=str, default='vgg',
                       help='extractor in listener: [vgg, ds2] (default: vgg)')
    group.add_argument('--activation', '-activation',
                       type=str, default='hardtanh',
                       help='activation function in listener`s cnn: [hardtanh, relu, elu] (default: hardtanh)')
    group.add_argument('--attn_mechanism', '-attn_mechanism',
                       type=str, default='loc',
                       help='option to specify the attention mechanism method')
    group.add_argument('--teacher_forcing_ratio', '-teacher_forcing_ratio',
                       type=float, default=0.99,
                       help='teacher forcing ratio in decoding (default: 0.99)')


def build_train_opts(parser):
    """ Training and saving options """
    group = parser.add_argument_group('General')
    group.add_argument('--dataset_path', '-dataset_path',
                       type=str, default='/data1/',
                       help='path of dataset')
    group.add_argument('--data_list_path', '-data_list_path',
                       type=str, default='./data/data_list/filter_train_list.csv',
                       help='list of training / test set')
    group.add_argument('--label_path', '-label_path',
                       type=str, default='./data/label/aihub_labels.csv',
                       help='path of character labels')
    group.add_argument('--init_uniform', '-init_uniform',
                       action='store_true', default=False,
                       help='flag indication whether to initiate model`s parameters as uniformly')
    group.add_argument('--spec_augment', '-spec_augment',
                       action='store_true', default=False,
                       help='flag indication whether to use spec augmentation or not')
    group.add_argument('--noise_augment', '-noise_augment',
                       action='store_true', default=False,
                       help='flag indication whether to use noise augmentation or not')
    group.add_argument('--noiseset_size', '-noiseset_size',
                       type=int, default=1000,
                       help='size of noise dataset for noise augmentation (default: 1000)')
    group.add_argument('--noise_level', '-noise_level',
                       type=float, default=0.7,
                       help='set level of noise (default: 0.7)')
    group.add_argument('--use_cuda', '-use_cuda',
                       action='store_true', default=False,
                       help='flag indication whether to use cuda or not')
    group.add_argument('--batch_size', '-batch_size',
                       type=int, default=32,
                       help='batch size in training (default: 32)')
    group.add_argument('--num_workers', '-num_workers',
                       type=int, default=4,
                       help='number of workers in dataset loader (default: 4)')
    group.add_argument('--num_epochs', '-num_epochs',
                       type=int, default=20,
                       help='number of epochs in training (default: 20)')
    group.add_argument('--init_lr', '-init_lr',
                       type=float, default=1e-15,
                       help='initial learning rate => before ramp up lr (default: 1e-15)')
    group.add_argument('--high_plateau_lr', '-high_plateau_lr',
                       type=float, default=3e-04,
                       help='high plateau learning rate => after rampup lr (default: 3e-04)')
    group.add_argument('--low_plateau_lr', '-low_plateau_lr',
                       type=float, default=1e-05,
                       help='low plateau learning rate => after exponential decay (default: 1e-05)')
    group.add_argument('--valid_ratio', '-valid_ratio',
                       type=float, default=0.01,
                       help='validation dataset ratio in training dataset (default: 0.01)')
    group.add_argument('--max_len', '-max_len',
                       type=int, default=151,
                       help='maximum characters of sentence (default: 151)')
    group.add_argument('--max_grad_norm', '-max_grad_norm',
                       type=int, default=400,
                       help='value used for gradient norm clipping (default: 400)')
    group.add_argument('--rampup_period', '-rampup_period',
                       type=int, default=1000,
                       help='timestep of learning rate rampup (default: 1000)')
    group.add_argument('--decay_threshold', '-decay_threshold',
                       type=float, default=0.02,
                       help='If the improvement of cer less than this, exponential decay lr start. (default: 0.02)')
    group.add_argument('--exp_decay_period', '-exp_decay_period',
                       type=int, default=160000,
                       help='Timestep of learning rate decay (default: 160000)')
    group.add_argument('--teacher_forcing_step', '-teacher_forcing_step',
                       type=float, default=0.05,
                       help='The value at which teacher forcing ratio will be reducing')
    group.add_argument('--min_teacher_forcing_ratio', '-min_teacher_forcing_ratio',
                       type=float, default=0.7,
                       help='The minimum value of teacher forcing ratio')
    group.add_argument('--seed', '-seed',
                       type=int, default=7,
                       help='random seed (default: 7)')
    group.add_argument('--save_result_every', '-save_result_every',
                       type=int, default=1000,
                       help='to determine whether to store training results every N timesteps (default: 1000)')
    group.add_argument('--checkpoint_every', '-checkpoint_every',
                       type=int, default=5000,
                       help='to determine whether to store training checkpoint every N timesteps (default: 5000)')
    group.add_argument('--print_every', '-print_every',
                       type=int, default=10,
                       help='to determine whether to store training progress every N timesteps (default: 10)')
    group.add_argument('--resume', '-resume',
                       action='store_true', default=False,
                       help='Indicates if training has to be resumed from the latest checkpoint')


def build_preprocess_opts(parser):
    """ Pre-processing options """
    group = parser.add_argument_group('Input')
    group.add_argument('--sample_rate', '-sample_rate',
                       type=int, default=16000,
                       help='sample rate (default: 16000)')
    group.add_argument('--window_size', '-window_size',
                       type=int, default=20,
                       help='Window size for spectrogram (default: 20ms)')
    group.add_argument('--stride', '-stride',
                       type=int, default=10,
                       help='Window stride for spectrogram (default: 10ms)')
    group.add_argument('--n_mels', '-n_mels',
                       type=int, default=80,
                       help='number of mel filter (default: 80)')
    group.add_argument('--normalize', '-normalize',
                       action='store_true', default=False,
                       help='flag indication whether to normalize spectrogram or not')
    group.add_argument('--del_silence', '-del_silence',
                       action='store_true', default=False,
                       help='flag indication whether to delete silence or not')
    group.add_argument('--input_reverse', '-input_reverse',
                       action='store_true', default=False,
                       help='flag indication whether to reverse input or not')
    group.add_argument('--feature_extract_by', '-feature_extract_by',
                       type=str, default='librosa',
                       help='which library to use for feature extraction: [librosa, torchaudio] (default: librosa)')
    group.add_argument('--time_mask_para', '-time_mask_para',
                       type=int, default=50,
                       help='Hyper Parameter for Time Masking to limit time masking length (default: 50)')
    group.add_argument('--freq_mask_para', '-freq_mask_para',
                       type=int, default=12,
                       help='Hyper Parameter for Freq Masking to limit freq masking length (default: 12)')
    group.add_argument('--time_mask_num', '-time_mask_num',
                       type=int, default=2,
                       help='how many time-masked area to make (default: 2)')
    group.add_argument('--freq_mask_num', '-freq_mask_num',
                       type=int, default=2,
                       help='how many freq-masked area to make (default: 2)')


def build_eval_opts(parser):
    """ inference options """
    group = parser.add_argument_group('Eval')
    group.add_argument('--dataset_path', '-dataset_path',
                       type=str, default='/data1/',
                       help='path of dataset')
    group.add_argument('--data_list_path', '-data_list_path',
                       type=str, default='./data/data_list/filter_train_list.csv',
                       help='list of training / test set')
    group.add_argument('--label_path', '-label_path',
                       type=str, default='./data/label/aihub_labels.csv',
                       help='path of character labels')
    group.add_argument('--num_workers', '-num_workers',
                       type=int, default=4,
                       help='number of workers in dataset loader (default: 4)')
    group.add_argument('--use_cuda', '-use_cuda',
                       action='store_true', default=False,
                       help='flag indication whether to use cuda or not')
    group.add_argument('--model_path', '-model_path',
                       type=str, default=None,
                       help='path to load models (default: None)')
    group.add_argument('--batch_size', '-batch_size',
                       type=int, default=1,
                       help='batch size in inference (default: 1)')
    group.add_argument('--k', '-k',
                       type=int, default=5,
                       help='size of beam (default: 5)')
    group.add_argument('--decode', '-decode',
                       type=str, default='greedy',
                       help='to determine whethre to search using [greedy, beam] (default: greedy)')
    group.add_argument('--print_every', '-print_every',
                       type=int, default=10,
                       help='to determine whether to store inference progress every N timesteps (default: 10')


def print_preprocess_opts(opt):
    """ Print preprocess options """
    logger.info('--mode: %s' % str(opt.mode))
    logger.info('--sample_rate: %s' % str(opt.sample_rate))
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
    logger.info('--attn_mechanism: %s' % str(opt.attn_mechanism))
    logger.info('--num_heads: %s' % str(opt.num_heads))
    logger.info('--label_smoothing: %s' % str(opt.label_smoothing))
    logger.info('--listener_layer_size: %s' % str(opt.listener_layer_size))
    logger.info('--speller_layer_size: %s' % str(opt.speller_layer_size))
    logger.info('--extractor: %s' % str(opt.extractor))
    logger.info('--activation: %s' % str(opt.activation))
    logger.info('--rnn_type: %s' % str(opt.rnn_type))
    logger.info('--teacher_forcing_ratio: %s' % str(opt.teacher_forcing_ratio))


def print_train_opts(opt):
    """ Print train options """
    logger.info('--dataset_path: %s' % str(opt.dataset_path))
    logger.info('--data_list_path: %s' % str(opt.data_list_path))
    logger.info('--label_path: %s' % str(opt.label_path))
    logger.info('--init_uniform: %s' % str(opt.init_uniform))
    logger.info('--spec_augment: %s' % str(opt.spec_augment))
    logger.info('--noise_augment: %s' % str(opt.noise_augment))
    logger.info('--noiseset_size: %s' % str(opt.noiseset_size))
    logger.info('--noise_level: %s' % str(opt.noise_level))
    logger.info('--use_cuda: %s' % str(opt.use_cuda))
    logger.info('--batch_size: %s' % str(opt.batch_size))
    logger.info('--num_workers: %s' % str(opt.num_workers))
    logger.info('--num_epochs: %s' % str(opt.num_epochs))
    logger.info('--init_lr: %s' % str(opt.init_lr))
    logger.info('--high_plateau_lr: %s' % str(opt.high_plateau_lr))
    logger.info('--low_plateau_lr: %s' % str(opt.low_plateau_lr))
    logger.info('--decay_threshold: %s' % str(opt.decay_threshold))
    logger.info('--rampup_period: %s' % str(opt.rampup_period))
    logger.info('--exp_decay_period: %s' % str(opt.exp_decay_period))
    logger.info('--valid_ratio: %s' % str(opt.valid_ratio))
    logger.info('--max_len: %s' % str(opt.max_len))
    logger.info('--max_grad_norm: %s' % str(opt.max_grad_norm))
    logger.info('--teacher_forcing_step: %s' % str(opt.teacher_forcing_step))
    logger.info('--min_teacher_forcing_ratio: %s' % str(opt.min_teacher_forcing_ratio))
    logger.info('--seed: %s' % str(opt.seed))
    logger.info('--save_result_every: %s' % str(opt.save_result_every))
    logger.info('--checkpoint_every: %s' % str(opt.checkpoint_every))
    logger.info('--print_every: %s' % str(opt.print_every))
    logger.info('--resume: %s' % str(opt.resume))


def print_eval_opts(opt):
    """ Print evaltation options """
    logger.info('--dataset_path: %s' % str(opt.dataset_path))
    logger.info('--data_list_path: %s' % str(opt.data_list_path))
    logger.info('--label_path: %s' % str(opt.label_path))
    logger.info('--num_workers: %s' % str(opt.num_workers))
    logger.info('--use_cuda: %s' % str(opt.use_cuda))
    logger.info('--model_path: %s' % str(opt.model_path))
    logger.info('--batch_size: %s' % str(opt.batch_size))
    logger.info('--decode: %s' % str(opt.decode))
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
        print_eval_opts(opt)

    else:
        raise ValueError("Unsupported mode: {0}".format(mode))
