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

import Levenshtein as Lev
import time
from loader.loader import *
from definition import *
import math
import random

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()
    begin = epoch_begin = time.time()
    while True:
        feats, scripts, feat_lengths, script_lengths = queue.get()
        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1
            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue
        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)
        src_len = scripts.size(1)
        target = scripts[:, 1:]
        model.module.flatten_parameters()

        # Seq2seq forward()
        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
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
            # ====테스트
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

        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length


train.cumulative_batch_count = 0


def evaluate(model, dataloader, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length


def split_dataset(hparams, audio_paths, label_paths, valid_ratio=0.05, target_dict = dict()):
    """
    Dataset split into training and validation Dataset.
    Args:
        valid_ratio: ratio for validation data
    Inputs: hparams, audio_paths, label_paths, target_dict
        - **hparams**: set of hyper parameters
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
        - **target_dict**: dictionary of filename and labels
                {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    Local Variables:
        - **train_num**: num of training data
        - **batch_num**: total num of batch
        - **valid_batch_num**: num of batch for validation
        - **train_num_per_worker**: num of train data per CPU core
        - **data_paths**: temp variables for audio_paths and label_paths to be shuffled in the same order
        - **train_begin_idx**: begin index of worker`s training dataset
        - **train_end_idx**: end index of worker`s training dataset
    Outputs: train_batch_num, train_dataset, valid_dataset
        - **train_batch_num**: num of batch for training
        - **train_dataset**: list of training data
        - **valid_dataset**: list of validation data
    """
    train_dataset = list()
    train_num = math.ceil(len(audio_paths) * (1 - valid_ratio))
    batch_num = math.ceil(len(audio_paths) / hparams.batch_size)
    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num
    train_num_per_worker = math.ceil(train_num / hparams.workers)

    # generate random index list
    data_paths = list(zip(audio_paths, label_paths))
    random.shuffle(data_paths)
    audio_paths[:], label_paths[:] = zip(*data_paths)

    # train dataset을 worker 수 만큼 분리
    for idx in range(hparams.workers):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)
        train_dataset.append(BaseDataset(audio_paths[train_begin_idx:train_end_idx],
                                         label_paths[train_begin_idx:train_end_idx],
                                         SOS_token, EOS_token, target_dict))
    valid_dataset = BaseDataset(audio_paths[train_num:], label_paths[train_num:], SOS_token, EOS_token, target_dict)

    return train_batch_num, train_dataset, valid_dataset