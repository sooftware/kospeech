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

import torch
import random
from utils.distance import get_distance
from utils.define import logger

def evaluate(model, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_distance = 0
    total_length = 0
    total_sentence_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            y_hat, logit = model(feats, scripts, teacher_forcing_ratio=0.0, use_beam_search = False)
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            distance, length = get_distance(target, y_hat)
            total_distance += distance
            total_length += length
            total_sentence_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_distance / total_length