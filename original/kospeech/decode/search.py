# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import torch.nn as nn
from queue import Queue
from kospeech.utils import logger
from kospeech.metrics import (
    CharacterErrorRate,
    WordErrorRate
)
from kospeech.models import (
    DeepSpeech2,
    ListenAttendSpell,
    SpeechTransformer,
    Jasper,
    Conformer,
)


class GreedySearch(object):
    """ Provides greedy search and save result to csv format """
    def __init__(self, vocab, metric: str = 'char'):
        self.target_list = list()
        self.predict_list = list()
        self.vocab = vocab

        if metric == 'char':
            self.metric = CharacterErrorRate(vocab)
        elif metric == 'word':
            self.metric = WordErrorRate(vocab)
        else:
            raise ValueError("Unsupported metric : {0}".format(metric))

    def search(self, model: nn.Module, queue: Queue, device: str, print_every: int) -> float:
        cer = 0
        total_sent_num = 0
        timestep = 0

        if isinstance(model, nn.DataParallel):
            model = model.module
            if isinstance(model, ListenAttendSpell):
                architecture = 'las'
            elif isinstance(model, SpeechTransformer):
                architecture = 'transformer'
            elif isinstance(model, DeepSpeech2):
                architecture = 'deepspeech2'
            elif isinstance(model, Jasper):
                architecture = 'jasper'
            elif isinstance(model, Conformer):
                architecture = 'conformer'
            else:
                raise ValueError("Unsupported model : {0}".format(type(model)))
        else:
            if isinstance(model, ListenAttendSpell):
                architecture = 'las'
            elif isinstance(model, SpeechTransformer):
                architecture = 'transformer'
            elif isinstance(model, DeepSpeech2):
                architecture = 'deepspeech2'
            elif isinstance(model, Jasper):
                architecture = 'jasper'
            elif isinstance(model, Conformer):
                architecture = 'conformer'
            else:
                raise ValueError("Unsupported model : {0}".format(type(model)))

        model.eval()
        model.to(device)

        while True:
            inputs, targets, input_lengths, target_lengths = queue.get()
            if inputs.shape[0] == 0:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            y_hats = model.greedy_search(inputs, input_lengths, device)

            for idx in range(targets.size(0)):
                self.target_list.append(
                    self.vocab.label_to_string(targets[idx])
                )
                self.predict_list.append(
                    self.vocab.label_to_string(y_hats[idx].cpu().detach().numpy())
                )

            cer = self.metric(targets[:, 1:], y_hats)
            total_sent_num += targets.size(0)

            if timestep % print_every == 0:
                logger.info('cer: {:.2f}'.format(cer))

            timestep += 1

        return cer

    def save_result(self, save_path: str) -> None:
        results = {
            'targets': self.target_list,
            'predictions': self.predict_list
        }
        results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, encoding='cp949')


class BeamSearch(GreedySearch):
    """ Provides beam search decoding. """
    def __init__(self, vocab, k):
        super(BeamSearch, self).__init__(vocab)
        self.k = k

    def search(self, model: nn.Module, queue: Queue, device: str, print_every: int) -> float:
        if isinstance(model, nn.DataParallel):
            topk_decoder = TopKDecoder(model.module.decoder, self.k)
            model.module.set_decoder(topk_decoder)
        else:
            topk_decoder = TopKDecoder(model.decoder, self.k)
            model.set_decoder(topk_decoder)
        return super(BeamSearch, self).search(model, queue, device, print_every)
