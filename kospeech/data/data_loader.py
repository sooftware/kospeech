# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import threading
import torch
import random
from argparse import ArgumentParser
from torch.utils.data import Dataset
from kospeech import (
    load_dataset,
    logger,
    SpectrogramParser
)


class SpectrogramDataset(Dataset, SpectrogramParser):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        opt (ArgumentParser): set of arguments
        dataset_path (str): path of dataset
    """
    def __init__(
            self,
            audio_paths: list,              # list of audio paths
            transcripts: list,              # list of transcript paths
            sos_id: int,                    # identification of start of sequence token
            eos_id: int,                    # identification of end of sequence token
            opt: ArgumentParser,            # set of arguments
            spec_augment: bool = False,     # flag indication whether to use spec-augmentation of not
            dataset_path: str = None,       # path of dataset,
            audio_extension: str = 'pcm'    # audio extension
    ) -> None:
        super(SpectrogramDataset, self).__init__(
            feature_extract_by=opt.feature_extract_by, sample_rate=opt.sample_rate, n_mels=opt.n_mels,
            frame_length=opt.frame_length, frame_shift=opt.frame_shift, del_silence=opt.del_silence,
            input_reverse=opt.input_reverse, normalize=opt.normalize, freq_mask_para=opt.freq_mask_para,
            time_mask_num=opt.time_mask_num, freq_mask_num=opt.freq_mask_num,
            sos_id=sos_id, eos_id=eos_id,
            dataset_path=dataset_path, transform_method=opt.transform_method, audio_extension=audio_extension
        )
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.augment_methods = [self.VANILLA] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self.augment(spec_augment)
        self.shuffle()

    def get_item(self, idx):
        """ get feature vector & transcript """
        feature_vector = self.parse_audio(os.path.join(self.dataset_path, self.audio_paths[idx]), self.augment_methods[idx])
        transcript = self.parse_transcript(self.transcripts[idx])

        return feature_vector, transcript

    def parse_transcript(self, transcript):
        """ Parses transcript """
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript

    def augment(self, spec_augment):
        """ Spec Augmentation """
        if spec_augment:
            logger.info("Applying Spec Augmentation...")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.SPEC_AUGMENT)
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

    def shuffle(self):
        """ Shuffle dataset """
        tmp = list(zip(self.audio_paths, self.transcripts, self.augment_methods))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


class AudioDataLoader(threading.Thread):
    """
    Audio Data Loader

    Args:
        dataset (kodpeech.data.data_loader.SpectrogramDataset): dataset for feature & transcript matching
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        thread_id (int): identification of thread
    """
    def __init__(self, dataset, queue, batch_size, thread_id, pad_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id
        self.pad_id = pad_id

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)

        seq_lengths = list()
        target_lengths = list()

        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        """ Load data from MelSpectrogramDataset """
        logger.debug('loader %d start' % self.thread_id)

        while True:
            items = list()

            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break

                feature_vector, transcript = self.dataset.get_item(self.index)

                if feature_vector is not None:
                    items.append((feature_vector, transcript))

                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            batch = self.collate_fn(items, self.pad_id)
            self.queue.put(batch)

        logger.debug('loader %d stop' % self.thread_id)

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)


def _collate_fn(batch, pad_id):
    """ functions that pad to the maximum sequence length """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths


class MultiDataLoader(object):
    """
    Multi Data Loader using Threads.

    Args:
        dataset_list (list): list of MelSpectrogramDataset
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        num_workers (int): the number of cpu cores used
    """
    def __init__(self, dataset_list, queue, batch_size, num_workers, pad_id):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader = list()

        for idx in range(self.num_workers):
            self.loader.append(AudioDataLoader(self.dataset_list[idx], self.queue, self.batch_size, idx, pad_id))

    def start(self):
        """ Run threads """
        for idx in range(self.num_workers):
            self.loader[idx].start()

    def join(self):
        """ Wait for the other threads """
        for idx in range(self.num_workers):
            self.loader[idx].join()


def split_dataset(opt, transcripts_path, vocab):
    """
    split into training set and validation set.

    Args:
        opt (ArgumentParser): set of options
        transcripts_path (str): path of  transcripts

    Returns: train_batch_num, train_dataset_list, valid_dataset
        - **train_time_step** (int): number of time step for training
        - **trainset_list** (list): list of training dataset
        - **validset** (data_loader.MelSpectrogramDataset): validation dataset
    """
    logger.info("split dataset start !!")
    trainset_list = list()

    if opt.dataset == 'kspon':
        train_num = 620000
        valid_num = 2545
    elif opt.dataset == 'libri':
        train_num = 281241
        valid_num = 5567
    else:
        raise NotImplementedError("Unsupported Dataset : {0}".format(opt.dataset))

    audio_paths, transcripts = load_dataset(transcripts_path)

    total_time_step = math.ceil(len(audio_paths) / opt.batch_size)
    valid_time_step = math.ceil(valid_num / opt.batch_size)
    train_time_step = total_time_step - valid_time_step

    train_audio_paths = audio_paths[:train_num + 1]
    train_transcripts = transcripts[:train_num + 1]

    valid_audio_paths = audio_paths[train_num + 1:]
    valid_transcripts = transcripts[train_num + 1:]

    if opt.spec_augment:
        train_time_step <<= 1

    train_num_per_worker = math.ceil(train_num / opt.num_workers)

    # audio_paths & script_paths shuffled in the same order
    # for seperating train & validation
    tmp = list(zip(train_audio_paths, train_transcripts))
    random.shuffle(tmp)
    train_audio_paths, train_transcripts = zip(*tmp)

    # seperating the train dataset by the number of workers
    for idx in range(opt.num_workers):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)

        trainset_list.append(
            SpectrogramDataset(
                train_audio_paths[train_begin_idx:train_end_idx],
                train_transcripts[train_begin_idx:train_end_idx],
                vocab.sos_id, vocab.eos_id,
                opt=opt,
                spec_augment=opt.spec_augment,
                dataset_path=opt.dataset_path,
                audio_extension=opt.audio_extension
            )
        )

    validset = SpectrogramDataset(
        audio_paths=valid_audio_paths,
        transcripts=valid_transcripts,
        sos_id=vocab.sos_id, eos_id=vocab.eos_id,
        opt=opt, spec_augment=False,
        dataset_path=opt.dataset_path,
        audio_extension=opt.audio_extension
    )

    logger.info("split dataset complete !!")
    return train_time_step, trainset_list, validset
