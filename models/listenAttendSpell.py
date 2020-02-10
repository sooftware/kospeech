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

import torch.nn as nn
import torch.nn.functional as F

class ListenAttendSpell(nn.Module):
    """
    Listen, Attend and Spell (LAS) Model

    Args:
        - **listener**: encoder of seq2seq
        - **speller**: decoder of seq2seq
        - **decode_function**: A function used to generate symbols from RNN hidden state

    Reference:
        「Listen, Attend and Spell」 paper :  https://arxiv.org/abs/1508.01211
    """
    def __init__(self, listener, speller, decode_function = F.log_softmax, use_pyramidal = False):
        super(ListenAttendSpell, self).__init__()
        self.listener = listener
        self.speller = speller
        self.decode_function = decode_function
        self.use_pyramidal = use_pyramidal

    def flatten_parameters(self):
        if self.use_pyramidal:
            self.listener.bottom_rnn.flatten_parameters()
            self.listener.middle_rnn.flatten_parameters()
            self.listener.top_rnn.flatten_parameters()
        else:
            self.listener.rnn.flatten_parameters()
        self.speller.rnn.flatten_parameters()

    def beam_search(self, use=True):
        self.speller.use_beam_search = use

    # feats, labels, teacher_forcing_ratio
    def forward(self, feats, targets=None, teacher_forcing_ratio=0.99):
        listener_outputs, listener_hidden = self.listener(feats)
        y_hat, logit = self.speller(inputs = targets,
                              listener_hidden = listener_hidden,
                              listener_outputs = listener_outputs,
                              function = self.decode_function,
                              teacher_forcing_ratio = teacher_forcing_ratio)
        return y_hat, logit