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
from definition import logger

class HyperParams():
    """
    Set of Hyperparameters
    Hyperparameters:
        - **use_bidirectional**: if True, becomes a bidirectional encoder
        - **use_attention**: flag indication whether to use attention mechanism or not
        - **input_reverse**: flag indication whether to reverse input feature or not
        - **use_pickle**: flag indication whether to load data from pickle or not
        - **use_augment**: flag indication whether to use spec-augmentation or not
        - **augment_ratio**: ratio of spec-augmentation applied data
        - **encoder_layer_size**: num of encoder`s RNN cell
        - **decoder_layer_size**: num of decoder`s RNN cell
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
    """
    def __init__(self):
        self.use_bidirectional = True
        self.use_attention = True
        self.input_reverse = True
        self.use_augmentation = False
        self.use_pickle = True
        self.augment_ratio = 0.3
        self.hidden_size = 256
        self.dropout = 0.5
        self.encoder_layer_size = 5
        self.decoder_layer_size = 3
        self.batch_size = 6
        self.worker_num = 1
        self.max_epochs = 40
        self.lr = 0.0001
        self.teacher_forcing = 0.99
        self.seed = 1
        self.max_len = 80
        self.no_cuda = True
        self.save_name = 'model'
        self.mode = 'train'

    def input_params(self):
        use_bidirectional = input("use bidirectional : ")
        if use_bidirectional.lower() == 'true': self.use_bidirectional = True
        else: self.use_bidirectional = False
        use_attention = input("use_attention : ")
        if use_attention.lower() == 'true': self.use_attention = True
        else: self.use_attention = False
        use_pickle = input("use_pickle : ")
        if use_pickle.lower() == 'true': self.use_pickle = True
        else: self.use_pickle = False
        use_augmentation = input("use_augmentation : ")
        if use_augmentation.lower() == 'true': self.use_augmentation = True
        else: self.use_augmentation = False
        input_reverse = input("input reverse : ")
        if input_reverse.lower() == 'true' : self.input_reverse = True
        else: self.input_reverse = False
        self.hidden_size = int(input("hidden_size : "))
        self.dropout = float(input("dropout : "))
        self.encoder_layer_size = int(input("encoder_layer_size : "))
        self.decoder_layer_size = int(input("decoder_layer_size : "))
        self.batch_size = int(input("batch_size : "))
        self.worker_num = int(input("workers : "))
        self.max_epochs = int(input("max_epochs : "))
        self.lr = float(input("learning rate : "))
        self.teacher_forcing = float(input("teacher_forcing : "))
        self.augment_ratio = float(input("augment_ratio : "))
        self.seed = int(input("seed : "))

    def log_hparams(self):
        logger.info("use_bidirectional : %s" % str(self.use_bidirectional))
        logger.info("use_attention : %s" % str(self.use_attention))
        logger.info("use_pickle : %s" % str(self.use_pickle))
        logger.info("use_augmentation : %s" % str(self.use_augmentation))
        logger.info("augment_ratio : %0.2f" % self.augment_ratio)
        logger.info("input_reverse : %s" % str(self.input_reverse))
        logger.info("hidden_size : %d" % self.hidden_size)
        logger.info("encoder_layer_size : %d" % self.encoder_layer_size)
        logger.info("decoder_layer_size : %d" % self.decoder_layer_size)
        logger.info("dropout : %0.2f" % self.dropout)
        logger.info("batch_size : %d" % self.batch_size)
        logger.info("worker_num : %d" % self.worker_num)
        logger.info("max_epochs : %d" % self.max_epochs)
        logger.info("learning rate : %0.4f" % self.lr)
        logger.info("teacher_forcing_ratio : %0.2f" % self.teacher_forcing)
        logger.info("seed : %d" % self.seed)
        logger.info("max_len : %d" % self.max_len)
        logger.info("no_cuda : %s" % str(self.no_cuda))
        logger.info("save_name : %s" % self.save_name)
        logger.info("mode : %s" % self.mode)