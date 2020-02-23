"""
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
from utils.define import logger

class HyperParams():
    """
    Set of Hyperparameters
    Hyperparameters:
        - **use_bidirectional**: if True, becomes a bidirectional listener
        - **use_attention**: flag indication whether to use attention mechanism or not
        - **input_reverse**: flag indication whether to reverse input feature or not
        - **use_pickle**: flag indication whether to load data from pickle or not
        - **use_augment**: flag indication whether to use spec-augmentation or not
        - **augment_ratio**: ratio of spec-augmentation applied data
        - **listener_layer_size**: num of listener`s RNN cell
        - **speller_layer_size**: num of speller`s RNN cell
        - **hidden_size**: size of hidden state of RNN
        - **dropout**: dropout probability
        - **batch_size**: mini-batch size
        - **worker_num**: num of cpu core will be used
        - **max_epochs**: max epoch
        - **lr**: learning rate
        - **teacher_forcing**: The probability that teacher forcing will be used
        - **seed**: seed for random
        - **max_len**: a maximum allowed length for the sequence to be processed
        - **no_cuda**: if True, don`t use CUDA
        - **save_name**: save name of model
        - **mode**: train or test
        - **load_model**: flag indication whether to load weight file or not
        - **model_path**: path for weight file
    """
    def __init__(self,
                 use_bidirectional = True,
                 use_attention = True,
                 input_reverse = True,
                 use_augment = True,
                 use_pickle = True,
                 use_pyramidal = True,
                 use_cuda = True,
                 score_function = 'dot-product',
                 augment_ratio = 1.0,
                 hidden_size = 256,
                 dropout = 0.5,
                 listener_layer_size = 6,
                 speller_layer_size = 3,
                 batch_size = 6,
                 worker_num = 1,
                 max_epochs = 40,
                 use_multistep_lr = False,
                 init_lr = 0.0001,
                 high_plateau_lr = 0.001,
                 low_plateau_lr = 0.00003,
                 teacher_forcing = 0.9,
                 seed = 1,
                 max_len = 120,
                 mode = 'train',
                 load_model = False,
                 model_path = "nothing.pt"
                 ):
        self.use_bidirectional = use_bidirectional
        self.use_attention = use_attention
        self.input_reverse = input_reverse
        self.use_augment = use_augment
        self.use_pickle = use_pickle
        self.use_pyramidal = use_pyramidal
        self.use_cuda = use_cuda
        self.score_function = score_function
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
        self.mode = mode
        self.load_model = load_model
        self.model_path = model_path

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
        logger.info("high plateau learning rate : %0.4f" % self.high_plateau_lr)
        logger.info("low plateau learning rate : %0.4f" % self.low_plateau_lr)
        logger.info("teacher_forcing_ratio : %0.2f" % self.teacher_forcing)
        logger.info("seed : %d" % self.seed)
        logger.info("max_len : %d" % self.max_len)
        logger.info("use_cuda : %s" % str(self.use_cuda))
        logger.info("mode : %s" % self.mode)