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
class HyperParams():
    def __init__(self):
        self.bidirectional = False
        self.attention = False
        self.hidden_size = 64
        self.dropout = 0.5
        self.encoder_layer_size = 2
        self.decoder_layer_size = 1
        self.batch_size = 1
        self.workers = 4
        self.max_epochs = 30
        self.lr = 0.0001
        self.teacher_forcing = 0.99
        self.seed = 1
        self.max_len = 80
        self.no_cuda = True
        self.save_name = 'model'
        self.mode = 'train'

    def input_params(self):
        use_bidirectional = input("use bidirectional : ")
        if use_bidirectional.lower() == 'true':
            self.bidirectional = True
        use_attention = input("use_attention : ")
        if use_attention.lower() == 'true':
            self.attention = True
        self.hidden_size = int(input("hidden_size : "))
        self.dropout = float(input("dropout : "))
        self.encoder_layer_size = int(input("encoder_layer_size : "))
        self.decoder_layer_size = int(input("decoder_layer_size : "))
        self.batch_size = int(input("batch_size : "))
        self.workers = int(input("workers : "))
        self.max_epochs = int(input("max_epochs : "))
        self.lr = float(input("learning rate : "))
        self.teacher_forcing = float(input("teacher_forcing : "))
        self.seed = int(input("seed : "))