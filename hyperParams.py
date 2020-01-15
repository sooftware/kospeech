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
    def __init__(self):
        self.use_bidirectional = True
        self.use_attention = True
        self.use_augment = True
        self.hidden_size = 256
        self.dropout = 0.5
        self.encoder_layer_size = 5
        self.decoder_layer_size = 3
        self.batch_size = 8
        self.worker_num = 4
        self.max_epochs = 40
        self.lr = 0.0001
        self.teacher_forcing = 0.99
        self.seed = 1
        self.max_len = 80
        self.no_cuda = False
        self.save_name = 'model'
        self.mode = 'train'
        self.input_reverse = True
        self.augment_ratio = 0.3

    def input_params(self):
        use_bidirectional = input("use bidirectional : ")
        if use_bidirectional.lower() == 'true': self.use_bidirectional = True
        else: self.use_bidirectional = False
        use_attention = input("use_attention : ")
        if use_attention.lower() == 'true': self.use_attention = True
        else: self.use_attention = False
        use_augment = input("use_augment : ")
        if use_augment.lower() == 'true': self.use_augment = True
        else: self.use_augment = False
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

    def print_hparams(self):
        print("use_bidirectional : ", str(self.use_bidirectional))
        print("use_attention : ", str(self.use_attention))
        print("use_augment : ", str(self.use_augment))
        print("augment_ratio : ", self.augment_ratio)
        print("input_reverse : ", str(self.input_reverse))
        print("hidden_size : ", self.hidden_size)
        print("encoder_layer_size : ", self.encoder_layer_size)
        print("decoder_layer_size : ", self.decoder_layer_size)
        print("dropout : ", self.dropout)
        print("batch_size :", self.batch_size)
        print("worker_num : ", self.worker_num)
        print("max_epochs : ", self.max_epochs)
        print("learning rate : ", self.lr)
        print("teacher_forcing_ratio : ", self.teacher_forcing)
        print("seed : ", self.seed)
        print("max_len : ", self.max_len)
        print("no_cuda : ", str(self.no_cuda))
        print("save_name : ", self.save_name)
        print("mode : ", self.mode)