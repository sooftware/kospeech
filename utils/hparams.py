#from utils.define import logger


class HyperParams():
    """
    Set of Hyperparameters

    Hyperparameters:
        - **use_bidirectional** (bool): if True, becomes a bidirectional listener (default: True)
        - **use_attention** (bool): flag indication whether to use attention mechanism or not (default: True)
        - **use_label_smooth** (bool): flagindication whether to use label smoothing or not (default: True)
        - **input_reverse** (bool): flag indication whether to reverse input feature or not (default: True)
        - **use_pickle** (bool): flag indication whether to load data from pickle or not (default: False)
        - **use_augment** (bool): flag indication whether to use spec-augmentation or not (default: True)
        - **use_pyramidal** (bool): flag indication whether to use pyramidal rnn in listener or not (default: True)
        - **use_multistep_lr** (bool): flag indication whether to use multistep leraning rate or not (default:False)
        - **augment_ratio** (float): ratio of spec-augmentation applied data (default: 1.0)
        - **listener_layer_size** (int): num of listener`s RNN cell (default: 6)
        - **speller_layer_size** (int): num of speller`s RNN cell (default: 3)
        - **hidden_size** (int): size of hidden state of RNN (default: 256)
        - **dropout** (float): dropout probability (default: 0.5)
        - **batch_size** (int): mini-batch size (default: 12)
        - **worker_num** (int): num of cpu core will be used (default: 1)
        - **max_epochs** (int): max epoch (default: 40)
        - **init_lr** (float): initial learning rate (default: 1e-4)
        - **high_plateau_lr** (float): maximum learning rate after the ramp up phase (default: -)
        - **low_plateau_lr** (float): Steps to be maintained at a certain number to avoid extremely slow learning (default: -)
        - **teacher_forcing** (float): The probability that teacher forcing will be used (default: 0.90)
        - **seed** (int): seed for random (default: 1)
        - **max_len** (int): a maximum allowed length for the sequence to be processed (default: 120)
        - **use_cuda** (bool): if True, use CUDA (default: True)
    """
    def __init__(self,
                 use_bidirectional = True,
                 use_attention = True,
                 use_label_smooth = True,
                 input_reverse = True,
                 use_augment = True,
                 use_pickle = False,
                 use_pyramidal = True,
                 use_cuda = True,
                 pack_by_length = True,
                 augment_ratio = 1.0,
                 hidden_size = 256,
                 dropout = 0.5,
                 listener_layer_size = 6,
                 speller_layer_size = 3,
                 batch_size = 8,
                 worker_num = 1,
                 max_epochs = 40,
                 use_multistep_lr = False,
                 init_lr = 0.0001,
                 high_plateau_lr = 0.001,
                 low_plateau_lr = 0.00003,
                 teacher_forcing = 0.9,
                 seed = 1,
                 max_len = 328
                 ):
        self.use_bidirectional = use_bidirectional
        self.use_attention = use_attention
        self.use_label_smooth = use_label_smooth
        self.input_reverse = input_reverse
        self.use_augment = use_augment
        self.use_pickle = use_pickle
        self.use_pyramidal = use_pyramidal
        self.use_cuda = use_cuda
        self.pack_by_length = pack_by_length
        self.augment_ratio = augment_ratio
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.listener_layer_size = listener_layer_size
        self.speller_layer_size = speller_layer_size
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.max_epochs = max_epochs
        self.use_multistep_lr = use_multistep_lr
        self.init_lr = init_lr
        if use_multistep_lr:
            self.high_plateau_lr = high_plateau_lr
            self.low_plateau_lr = low_plateau_lr
        self.teacher_forcing = teacher_forcing
        self.seed = seed
        self.max_len = max_len
"""
    def logger_hparams(self):
        logger.info("use_bidirectional : %s" % str(self.use_bidirectional))
        logger.info("use_attention : %s" % str(self.use_attention))
        logger.info("use_pickle : %s" % str(self.use_pickle))
        logger.info("use_augment : %s" % str(self.use_augment))
        logger.info("use_pyramidal : %s" % str(self.use_pyramidal))
        logger.info("attention : %s" % self.score_function)
        logger.info("augment_ratio : %0.2f" % self.augment_ratio)
        logger.info("input_reverse : %s" % str(self.input_reverse))
        logger.info("hidden_size : %d" % self.hidden_size)
        logger.info("listener_layer_size : %d" % self.listener_layer_size)
        logger.info("speller_layer_size : %d" % self.speller_layer_size)
        logger.info("dropout : %0.2f" % self.dropout)
        logger.info("batch_size : %d" % self.batch_size)
        logger.info("worker_num : %d" % self.worker_num)
        logger.info("max_epochs : %d" % self.max_epochs)
        logger.info("initial learning rate : %0.4f" % self.init_lr)
        if self.use_multistep_lr:
            logger.info("high plateau learning rate : %0.4f" % self.high_plateau_lr)
            logger.info("low plateau learning rate : %0.4f" % self.low_plateau_lr)
        logger.info("teacher_forcing_ratio : %0.2f" % self.teacher_forcing)
        logger.info("seed : %d" % self.seed)
        logger.info("max_len : %d" % self.max_len)
        logger.info("use_cuda : %s" % str(self.use_cuda))
"""