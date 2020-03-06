"""
Copyright 2020- Kai.Lib

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import time
from utils.distance import get_distance
from utils.define import logger, id2char, EOS_TOKEN
from utils.save import save_step_result
from train.lr import ramp_up, exp_decay
train_step_result = {'loss': [], 'cer': []}

def train(model, hparams, epoch, total_time_step, queue,
          criterion, optimizer, device, train_begin, worker_num,
          print_time_step=10, teacher_forcing_ratio=0.90):
    """
    Args:
        model (torch.nn): Model to be trained
        optimizer (torch.optim): optimizer for training
        teacher_forcing_ratio (float):  The probability that teacher forcing will be used (default: 0.90)
        print_time_step (int): Parameters to determine how many steps to output
        queue (queue): queue for threading
        criterion (torch.nn): loss function ex) nn.CrossEntropyLoss, LabelSmoothingLoss etc ..
        device (torch.cuda): device used ('cuda' or 'cpu')
        worker_num (int): the number of cpu cores used

    Returns:
        - **cer** (float): character error rate
        - **loss** (float): loss of present epoch
    """
    total_loss = 0.
    total_num = 0
    total_distance = 0
    total_length = 0
    total_sent_num = 0
    time_step = 0

    model.train()
    begin = epoch_begin = time.time()

    while True:
        if hparams.use_multistep_lr and epoch == 0 and time_step < 1000:
            ramp_up(optimizer, time_step, hparams)
        if hparams.use_multistep_lr and epoch == 1:
            exp_decay(optimizer, total_time_step, hparams)
        feats, targets, feat_lengths, label_lengths = queue.get()
        if feats.shape[0] == 0:
            # empty feats means closing one loader
            worker_num -= 1
            logger.debug('left train_loader: %d' % (worker_num))

            if worker_num == 0:
                break
            else:
                continue
        optimizer.zero_grad()

        feats = feats.to(device)
        targets = targets.to(device)
        target = targets[:, 1:]
        model.module.flatten_parameters()

        y_hat, logit = model(feats, targets, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))

        total_loss += loss.item()
        total_num += sum(feat_lengths)
        distance, length = get_distance(target, y_hat, id2char, EOS_TOKEN)
        total_distance += distance
        total_length += length
        total_sent_num += target.size(0)
        loss.backward()
        optimizer.step()

        if time_step % print_time_step == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'.format(
                time_step,
                total_time_step,
                total_loss / total_num,
                total_distance / total_length,
                elapsed, epoch_elapsed, train_elapsed)
            )
            begin = time.time()

        if time_step % 1000 == 0:
            save_step_result(train_step_result, total_loss / total_num, total_distance / total_length)

        time_step += 1
        train.cumulative_batch_count += 1

    loss = total_loss / total_num
    cer = total_distance / total_length
    logger.info('train() completed')
    return loss, cer

train.cumulative_batch_count = 0