import math
import threading
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from kospeech.data.label_loader import load_targets
from kospeech.data.preprocess.parser import SpeechParser
from kospeech.utils import logger, PAD_token, SOS_token, EOS_token


class SpeechDataset(Dataset, SpeechParser):
    """
    Dataset for mel-spectrogram & transcript matching

    Args:
        audio_paths (list): set of audio path
        script_paths (list): set of script paths
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        target_dict (dict): dictionary of filename and labels
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        noise_augment (bool): flag indication whether to use noise-augmentation or not (default: True)
        opt (ArgumentParser): set of arguments
    """
    def __init__(self, audio_paths, script_paths, sos_id, eos_id, target_dict, opt, spec_augment=False,
                 noise_augment=False, dataset_path=None, noiseset_size=0, noise_level=0.7):
        super(SpeechDataset, self).__init__(feature_extract_by=opt.feature_extract_by, sample_rate=opt.sample_rate,
                                            n_mels=opt.n_mels, window_size=opt.window_size, stride=opt.stride,
                                            del_silence=opt.del_silence, input_reverse=opt.input_reverse,
                                            normalize=opt.normalize, target_dict=target_dict,
                                            time_mask_para=opt.time_mask_para, freq_mask_para=opt.freq_mask_para,
                                            time_mask_num=opt.time_mask_num, freq_mask_num=opt.freq_mask_num,
                                            sos_id=sos_id, eos_id=eos_id, dataset_path=dataset_path,
                                            noiseset_size=noiseset_size, noise_level=noise_level, noise_augment=noise_augment)
        self.audio_paths = list(audio_paths)
        self.script_paths = list(script_paths)
        self.augment_methods = [self.VANILLA] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self.augmentation(spec_augment, noise_augment)
        self.shuffle()

    def get_item(self, idx):
        """ get spectrogram & label """
        transcript = self.parse_transcript(self.script_paths[idx])
        spectrogram = self.parse_audio(self.audio_paths[idx], self.augment_methods[idx])

        if spectrogram is None:
            return None, None
        else:
            return spectrogram, transcript

    def parse_transcript(self, script_path):
        """ Parses scripts @Override """
        transcripts = list()

        key = script_path.split('/')[-1].split('.')[0]
        transcript = self.target_dict[key]

        tokens = transcript.split(' ')

        transcripts.append(int(self.sos_id))
        for token in tokens:
            transcripts.append(int(token))
        transcripts.append(int(self.eos_id))

        return transcripts

    def augmentation(self, spec_augment, noise_augment):
        """ Spec & Noise Augmentation """
        if spec_augment:
            logger.info("Applying Spec Augmentation...")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.SPEC_AUGMENT)
                self.audio_paths.append(self.audio_paths[idx])
                self.script_paths.append(self.script_paths[idx])

        if noise_augment:
            logger.info("Applying Noise Augmentation...")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.NOISE_INJECTION)
                self.audio_paths.append(self.audio_paths[idx])
                self.script_paths.append(self.script_paths[idx])

    def shuffle(self):
        """ Shuffle dataset """
        tmp = list(zip(self.audio_paths, self.script_paths, self.augment_methods))
        random.shuffle(tmp)
        self.audio_paths, self.script_paths, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


class AudioLoader(threading.Thread):
    """
    Audio Data Loader

    Args:
        dataset (e2e.data_loader.MelSpectrogramDataset): dataset for spectrogram & script matching
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
        """ Load data from MelSpectrogramDataset """
        logger.debug('loader %d start' % self.thread_id)

        while True:
            items = list()

            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break

                spectrogram, transcript = self.dataset.get_item(self.index)

                if spectrogram is not None:
                    items.append((spectrogram, transcript))

                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

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


class MultiAudioLoader(object):
    """
    Multi Data Loader using Threads.

    Args:
        dataset_list (list): list of MelSpectrogramDataset
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
            self.loader.append(AudioLoader(self.dataset_list[idx], self.queue, self.batch_size, idx))

    def start(self):
        """ Run threads """
        for idx in range(self.num_workers):
            self.loader[idx].start()

    def join(self):
        """ Wait for the other threads """
        for idx in range(self.num_workers):
            self.loader[idx].join()


def split_dataset(opt, audio_paths, script_paths):
    """
    split into training set and validation set.

    Args:
        opt (ArgumentParser): set of options
        audio_paths (list): set of audio path
        script_paths (list): set of script path

    Returns: train_batch_num, train_dataset_list, valid_dataset
        - **train_time_step** (int): number of time step for training
        - **trainset_list** (list): list of training dataset
        - **validset** (data_loader.MelSpectrogramDataset): validation dataset
    """
    target_dict = load_targets(script_paths)

    logger.info("split dataset start !!")
    trainset_list = list()
    train_num = math.ceil(len(audio_paths) * (1 - opt.valid_ratio))
    total_time_step = math.ceil(len(audio_paths) / opt.batch_size)
    valid_time_step = math.ceil(total_time_step * opt.valid_ratio)
    train_time_step = total_time_step - valid_time_step
    residual = train_time_step

    if opt.spec_augment:
        train_time_step += residual

    if opt.noise_augment:
        train_time_step += residual

    train_num_per_worker = math.ceil(train_num / opt.num_workers)

    # audio_paths & script_paths shuffled in the same order
    # for seperating train & validation
    tmp = list(zip(audio_paths, script_paths))
    random.shuffle(tmp)
    audio_paths, script_paths = zip(*tmp)

    # seperating the train dataset by the number of workers
    for idx in range(opt.num_workers):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)

        trainset_list.append(
            SpeechDataset(
                audio_paths[train_begin_idx:train_end_idx],
                script_paths[train_begin_idx:train_end_idx],
                SOS_token, EOS_token,
                target_dict=target_dict,
                opt=opt,
                spec_augment=opt.spec_augment,
                noise_augment=opt.noise_augment,
                dataset_path=opt.dataset_path,
                noiseset_size=opt.noiseset_size,
                noise_level=opt.noise_level
            )
        )

    validset = SpeechDataset(
        audio_paths=audio_paths[train_num:],
        script_paths=script_paths[train_num:],
        sos_id=SOS_token, eos_id=EOS_token,
        target_dict=target_dict,
        opt=opt,
        spec_augment=False,
        noise_augment=False
    )

    logger.info("split dataset complete !!")
    return train_time_step, trainset_list, validset


def load_data_list(data_list_path, dataset_path):
    """
    Provides set of audio path & label path

    Args:
        data_list_path (str): csv file with training or test data list path.
        dataset_path (str): dataset path.

    Returns: audio_paths, script_paths
        - **audio_paths** (list): set of audio path
        - **script_paths** (list): set of label path
    """
    data_list = pd.read_csv(data_list_path, "r", delimiter=",", encoding="cp949")
    audio_paths = list(dataset_path + data_list["audio"])
    script_paths = list(dataset_path + data_list["label"])

    return audio_paths, script_paths
