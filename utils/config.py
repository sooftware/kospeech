from utils.definition import logger


class Config:
    """
    Configuration

    Args:
        use_bidirectional (bool): if True, becomes a bidirectional listener (default: True)
        use_label_smooth (bool): flag indication whether to use label smoothing or not (default: True)
        input_reverse (bool): flag indication whether to reverse input feature or not (default: True)
        use_pickle (bool): flag indication whether to load data from pickle or not (default: False)
        use_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        augment_num (float): number of spec-augmentation applied per data (default: 2)
        listener_layer_size (int): num of listener`s RNN cell (default: 6)
        speller_layer_size (int): num of speller`s RNN cell (default: 3)
        hidden_dim (int): size of hidden state of RNN (default: 256)
        dropout (float): dropout probability (default: 0.5)
        batch_size (int): mini-batch size (default: 12)
        worker_num (int): num of cpu core will be used (default: 1)
        max_epochs (int): max epoch (default: 40)
        lr (float): initial learning rate (default: 1e-4)
        teacher_forcing (float): The probability that teacher forcing will be used (default: 0.90)
        seed (int): seed for random (default: 1)
        max_len (int): a maximum allowed length for the sequence to be processed (default: 120)
        use_cuda (bool): if True, use CUDA (default: True)
    """

    def __init__(self,
                 use_bidirectional=True,
                 use_label_smooth=True,
                 input_reverse=True,
                 use_augment=True,
                 use_pickle=False,
                 use_cuda=True,
                 augment_num=1,
                 hidden_dim=256,
                 dropout=0.5,
                 listener_layer_size=5,
                 speller_layer_size=3,
                 num_head=12,
                 attn_dim=64,
                 batch_size=32,
                 worker_num=1,
                 max_epochs=40,
                 lr=0.001,
                 teacher_forcing=0.99,
                 seed=1,
                 max_len=151,
                 load_model=False,
                 model_path=None,
                 n_mels=80,
                 sr=16000,
                 window_size=20,  # ms
                 stride=10,       # ms
                 save_result_every=1000,
                 save_model_every=10000,
                 print_every=10,
                 ):
        self.use_bidirectional = use_bidirectional
        self.use_label_smooth = use_label_smooth
        self.input_reverse = input_reverse
        self.use_augment = use_augment
        self.use_pickle = use_pickle
        self.use_cuda = use_cuda
        self.augment_num = augment_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.listener_layer_size = listener_layer_size
        self.speller_layer_size = speller_layer_size
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.max_epochs = max_epochs
        self.num_head = num_head
        self.attn_dim = attn_dim
        self.lr = lr
        self.teacher_forcing = teacher_forcing
        self.seed = seed
        self.max_len = max_len
        self.n_mels = n_mels
        self.sr = sr
        self.window_size = window_size
        self.stride = stride
        self.save_result_every = save_result_every
        self.save_model_every = save_model_every
        self.print_every = print_every
        self.load_model = load_model
        self.model_path = model_path
        self.print_log()

    def print_log(self):
        """ print information of configuration """
        logger.info("use_bidirectional : %s" % str(self.use_bidirectional))
        logger.info("use_pickle : %s" % str(self.use_pickle))
        logger.info("use_augment : %s" % str(self.use_augment))
        logger.info("augment_num : %d" % self.augment_num)
        logger.info("input_reverse : %s" % str(self.input_reverse))
        logger.info("hidden_dim : %d" % self.hidden_dim)
        logger.info("listener_layer_size : %d" % self.listener_layer_size)
        logger.info("speller_layer_size : %d" % self.speller_layer_size)
        logger.info("num_head : %d" % self.num_head)
        logger.info("attn_dim : %d" % self.attn_dim)
        logger.info("dropout : %0.2f" % self.dropout)
        logger.info("batch_size : %d" % self.batch_size)
        logger.info("worker_num : %d" % self.worker_num)
        logger.info("max_epochs : %d" % self.max_epochs)
        logger.info("initial learning rate : %0.4f" % self.lr)
        logger.info("teacher_forcing_ratio : %0.2f" % self.teacher_forcing)
        logger.info("seed : %d" % self.seed)
        logger.info("max_len : %d" % self.max_len)
        logger.info("use_cuda : %s" % str(self.use_cuda))
        logger.info("n_mels : %d" % self.n_mels)
        logger.info("sr : %d" % self.sr)
        logger.info("window_size : %d" % self.window_size)
        logger.info("stride : %s" % str(self.stride))
