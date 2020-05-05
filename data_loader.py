import math
import random
import pickle
import torch
import threading
import pandas as pd
from definition import logger, SOS_token, EOS_token, PAD_token
from torch.utils.data import Dataset
from feature import spec_augment, get_librosa_melspectrogram, get_torchaudio_melspectrogram
from utils import get_label, save_pickle

feature_extract_funtions = {
    'librosa': get_librosa_melspectrogram,
    'torchaudio': get_torchaudio_melspectrogram
}


class SpectrogramDataset(Dataset):
    """
    Dataset for audio & label matching

    Args:
        audio_paths (list): set of audio path
        label_paths (list): set of label paths
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        target_dict (dict): dictionary of filename and labels
        use_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        args (ArgumentParser): set of arguments
    """

    def __init__(self, audio_paths, label_paths, sos_id, eos_id, target_dict=None, args=None, use_augment=True):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.target_dict = target_dict
        self.augment_flags = [False] * len(self.audio_paths)
        self.get_feature = feature_extract_funtions[args.feature_extract_by]
        self.args = args
        if use_augment:
            self.augment_num = args.augment_num
            self.augmentation()
        self.shuffle()

    def get_item(self, idx):
        label = get_label(self.label_paths[idx], self.sos_id, self.eos_id, self.target_dict)
        spectrogram = self.get_feature(
            self.audio_paths[idx],
            n_mels=self.args.n_mels,
            input_reverse=self.args.input_reverse,
            del_silence=self.args.del_silence,
            normalize=self.args.normalize,
            sr=self.args.sr,
            window_size=self.args.window_size,
            stride=self.args.stride
        )

        if spectrogram is None:  # exception handling
            return None, None

        if self.augment_flags[idx]:
            spectrogram = spec_augment(
                spectrogram,
                time_mask_para=self.args.time_mask_para,
                freq_mask_para=self.args.freq_mask_para,
                time_mask_num=self.args.time_mask_num,
                freq_mask_num=self.args.freq_mask_num
            )

        return spectrogram, label

    def augmentation(self):
        augment_end_idx = int(0 + ((len(self.audio_paths) - 0) * self.augment_num))
        logger.info("Applying Augmentation...")

        for _ in range(self.augment_num):
            for idx in range(augment_end_idx):
                self.augment_flags.append(True)
                self.audio_paths.append(self.audio_paths[idx])
                self.label_paths.append(self.label_paths[idx])

    def shuffle(self):
        temp = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
        random.shuffle(temp)
        self.audio_paths, self.label_paths, self.augment_flags = zip(*temp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


def split_dataset(args, audio_paths, label_paths, target_dict=None):
    """
    split into training set and validation set.

    Args:
        args (utils.args.Arguments): set of arguments
        audio_paths (list): set of audio path
        label_paths (list): set of label path
        target_dict (dict): dictionary of filename and target

    Returns: train_batch_num, train_dataset_list, valid_dataset
        - **train_time_step** (int): number of time step for training
        - **trainset_list** (list): list of training dataset
        - **validset** (data_loader.SpectrogramDataset): validation dataset
    """
    logger.info("split dataset start !!")

    trainset_list = list()
    train_num = math.ceil(len(audio_paths) * (1 - args.valid_ratio))
    total_time_step = math.ceil(len(audio_paths) / args.batch_size)
    valid_time_step = math.ceil(total_time_step * args.valid_ratio)
    train_time_step = total_time_step - valid_time_step

    if args.use_augment:
        train_time_step = int(train_time_step * (1 + args.augment_num))

    train_num_per_worker = math.ceil(train_num / args.num_workers)

    # audio_paths & label_paths shuffled in the same order
    # for seperating train & validation
    data_paths = list(zip(audio_paths, label_paths))
    random.shuffle(data_paths)
    audio_paths, label_paths = zip(*data_paths)

    # seperating the train dataset by the number of workers
    for idx in range(args.num_workers):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)

        trainset_list.append(SpectrogramDataset(
            audio_paths=audio_paths[train_begin_idx:train_end_idx],
            label_paths=label_paths[train_begin_idx:train_end_idx],
            sos_id=SOS_token, eos_id=EOS_token,
            target_dict=target_dict,
            use_augment=args.use_augment,
            args=args
        ))

    validset = SpectrogramDataset(
        audio_paths=audio_paths[train_num:],
        label_paths=label_paths[train_num:],
        sos_id=SOS_token, eos_id=EOS_token,
        target_dict=target_dict,
        args=args,
        use_augment=False
    )

    save_pickle(trainset_list, './data/pickle/trainset_list')
    save_pickle(validset, './data/pickle/validset')

    logger.info("split dataset complete !!")
    return train_time_step, trainset_list, validset


class MultiLoader:
    """
    Multi Data Loader using Threads.

    Args:
        dataset_list (list): list of SpectrogramDataset
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        num_workers (int): the number of cpu cores used
    """
    def __init__(self, dataset_list, queue, batch_size, num_workers):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader = list()

        for idx in range(self.num_workers):
            self.loader.append(AudioDataLoader(self.dataset_list[idx], self.queue, self.batch_size, idx))

    def start(self):
        for idx in range(self.num_workers):
            self.loader[idx].start()

    def join(self):
        for idx in range(self.num_workers):
            self.loader[idx].join()


class AudioDataLoader(threading.Thread):
    """
    Audio Data Loader

    Args:
        dataset (data_loader.SpectrogramDataset): object of SpectrogramDataset
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        thread_id (int): identification of thread
    """
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)

        seq_lengths = list()
        target_lengths = list()

        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % self.thread_id)
        while True:
            items = list()

            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break

                feat, label = self.dataset.get_item(self.index)

                if feat is not None:
                    items.append((feat, label))

                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)

        logger.debug('loader %d stop' % self.thread_id)

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)


def _collate_fn(batch):
    """ functions that pad to the maximum sequence length """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)  # sort by sequence length
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
    targets.fill_(PAD_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths


def load_data_list(data_list_path, dataset_path):
    """
    Provides set of audio path & label path

    Args:
        data_list_path (str): csv file with training or test data list path.
        dataset_path (str): dataset path.

    Returns: audio_paths, label_paths
        - **audio_paths** (list): set of audio path
        - **label_paths** (list): set of label path
    """
    data_list = pd.read_csv(data_list_path, "r", delimiter=",", encoding="cp949")
    audio_paths = list(dataset_path + data_list["audio"])
    label_paths = list(dataset_path + data_list["label"])

    return audio_paths, label_paths


def load_pickle(filepath, message=""):
    """
    load pickle file

    Args:
        filepath (str): Path to pickle file to load
        message (str): message to print

    Returns: load_result
        -**load_result** : load result of pickle
    """
    with open(filepath, "rb") as f:
        load_result = pickle.load(f)
        logger.info(message)
        return load_result
