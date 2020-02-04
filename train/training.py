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
import time, torch, random
from train.distance import get_distance
from definition import logger

from train.save_and_load import save_step_result

train_step_result = {'loss': [], 'cer': []}

def train(model, total_batch_size, queue, loss_func, optimizer, device, train_begin, worker_num, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()
    begin = epoch_begin = time.time()

    while True:
        feats, targets, feat_lengths, label_lengths = queue.get()
        if feats.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % (worker_num))

            if worker_num == 0: break
            else: continue
        optimizer.zero_grad()

        feats = feats.to(device)
        targets = targets.to(device)
        target = targets[:, 1:]
        model.module.flatten_parameters()

        # Seq2seq forward()
        y_hat, logit = model(feats, targets, teacher_forcing_ratio)

        loss = loss_func(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths)
        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length
        total_sent_num += target.size(0)
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

        if batch % 1000 == 0:
            save_step_result(train_step_result, total_loss / total_num, total_dist / total_length)

        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length

train.cumulative_batch_count = 0