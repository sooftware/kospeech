import random
import math
from torch.utils.data import Dataset
from utils.feature import get_librosa_mfcc, spec_augment, get_librosa_melspectrogram
from utils.label import get_label, label_to_string
from utils.define import logger, SOS_TOKEN, EOS_TOKEN


class BaseDataset(Dataset):
    """
    Dataset for audio & label matching

    Args:
        audio_paths (list): set of audio path
        label_paths (list): set of label paths
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        target_dict (dict): dictionary of filename and labels
        input_reverse (bool): flag indication whether to reverse input feature or not (default: True)
        use_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        augment_ratio (float): ratio of spec-augmentation applied data (default: 1.0)
        pack_by_length (bool): pack by similar sequence length
        batch_size (int): mini batch size
    """
    def __init__(self, audio_paths, label_paths, sos_id = 2037, eos_id = 2038,
                 target_dict = None, input_reverse = True, use_augment = True,
                 batch_size = None, augment_ratio = 0.3, pack_by_length = True):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_size = batch_size
        self.target_dict = target_dict
        self.input_reverse = input_reverse
        self.augment_ratio = augment_ratio
        self.augment_flags = [False] * len(self.audio_paths)
        self.pack_by_length = pack_by_length
        if use_augment:
            self.augmentation()
        if pack_by_length:
            self.sort_by_length()
            self.audio_paths, self.label_paths, self.augment_flags = self.batch_shuffle(remain_drop=False)
        else:
            bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
            random.shuffle(bundle)
            self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def get_item(self, idx):
        label = get_label(self.label_paths[idx], sos_id = self.sos_id, eos_id = self.eos_id, target_dict = self.target_dict)
        feat = get_librosa_melspectrogram(self.audio_paths[idx], n_mels = 128, mel_type='log_mel', input_reverse = self.input_reverse)
        # exception handling
        if feat is None:
            return None, None
        if self.augment_flags[idx]:
            feat = spec_augment(feat, T = 70, F = 20, time_mask_num = 2, freq_mask_num = 2 )
        return feat, label

    def augmentation(self):
        """ Apply Spec-Augmentation """
        augment_end_idx = int(0 + ((len(self.audio_paths) - 0) * self.augment_ratio))
        logger.info("Applying Augmentation...")

        for idx in range(augment_end_idx):
            self.augment_flags.append(True)
            self.audio_paths.append(self.audio_paths[idx])
            self.label_paths.append(self.label_paths[idx])

    def shuffle(self):
        """ Shuffle Dataset """
        if self.pack_by_length:
            self.audio_paths, self.label_paths, self.augment_flags = self.batch_shuffle(remain_drop=False)
        else:
            bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
            random.shuffle(bundle)
            self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def sort_by_length(self):
        """ descending sort by sequence length """
        target_lengths = list()
        for idx, label_path in enumerate(self.label_paths):
            key = label_path.split('/')[-1].split('.')[0]
            target_lengths.append(len(self.target_dict[key].split()))

        bundle = list(zip(target_lengths, self.audio_paths, self.label_paths, self.augment_flags))
        junk, self.audio_paths, self.label_paths, self.augment_flags = zip(*sorted(bundle, reverse=True))


    def batch_shuffle(self, remain_drop = False):
        """ batch shuffle """
        total_audio_batch, total_label_batch, total_augment_flag = list(), list(), list()
        tmp_audio_batch, tmp_label_batch, tmp_augment_flag = list(), list(), list()
        index = 0

        while True:
            if index == len(self.audio_paths):
                if len(tmp_audio_batch) != 0:
                    total_audio_batch.append(tmp_audio_batch)
                    total_label_batch.append(tmp_label_batch)
                    total_augment_flag.append(tmp_augment_flag)
                break
            if len(tmp_audio_batch) == self.batch_size:
                total_audio_batch.append(tmp_audio_batch)
                total_label_batch.append(tmp_label_batch)
                total_augment_flag.append(tmp_augment_flag)
                tmp_audio_batch, tmp_label_batch, tmp_augment_flag = list(), list(), list()
            tmp_audio_batch.append(self.audio_paths[index])
            tmp_label_batch.append(self.label_paths[index])
            tmp_augment_flag.append(self.augment_flags[index])
            index += 1

        remain_audio, remain_label, remain_augment_flag = total_audio_batch[-1], total_label_batch[-1], total_augment_flag[-1]
        total_audio_batch, total_label_batch, total_augment_flag = total_audio_batch[:-1], total_label_batch[:-1], total_augment_flag[:-1]

        bundle = list(zip(total_audio_batch, total_label_batch, total_augment_flag))
        random.shuffle(bundle)
        total_audio_batch, total_label_batch, total_augment_flag = zip(*bundle)

        audio_paths = list()
        label_paths = list()
        augment_flags = list()

        for (audio_batch, label_batch, augment_flag) in zip(total_audio_batch, total_label_batch, total_augment_flag):
            audio_paths.extend(audio_batch)
            label_paths.extend(label_batch)
            augment_flags.extend(augment_flag)

        audio_paths = list(audio_paths)
        label_paths = list(label_paths)
        augment_flags = list(augment_flags)

        if not remain_drop:
            audio_paths.extend(remain_audio)
            label_paths.extend(remain_label)
            augment_flags.extend(remain_augment_flag)

        return audio_paths, label_paths, augment_flags


def split_dataset(hparams, audio_paths, label_paths, valid_ratio=0.05, target_dict = None):
    """
    Dataset split into training and validation Dataset.

    Args:
        hparams (utils.hparams.HyperParams): set of hyper parameters
        audio_paths (list): set of audio path
        label_paths (list): set of label path
        target_dict (dict): dictionary of filename and labels

    Returns: train_batch_num, train_dataset_list, valid_dataset
        - **train_batch_num** (int): num of batch for training
        - **train_dataset_list** (list): list of training dataset
        - **valid_dataset** (utils.dataset.BaseDataset): validation dataset
    """
    logger.info("split dataset start !!")
    train_dataset_list = list()
    train_num = math.ceil(len(audio_paths) * (1 - valid_ratio))
    total_time_step = math.ceil(len(audio_paths) / hparams.batch_size)
    valid_time_step = math.ceil(total_time_step * valid_ratio)
    train_time_step = total_time_step - valid_time_step
    if hparams.use_augment:
        train_time_step = int( train_time_step * (1 + hparams.augment_ratio))
    train_num_per_worker = math.ceil(train_num / hparams.worker_num)

    # audio_paths & label_paths shuffled in the same order
    # for seperating train & validation
    data_paths = list(zip(audio_paths, label_paths))
    random.shuffle(data_paths)
    audio_paths, label_paths = zip(*data_paths)

    # seperating the train dataset by the number of workers
    for idx in range(hparams.worker_num):
        train_begin_index = train_num_per_worker * idx
        train_end_index = min(train_num_per_worker * (idx + 1), train_num)
        train_dataset_list.append(BaseDataset(
                                    audio_paths=audio_paths[train_begin_index:train_end_index],
                                    label_paths=label_paths[train_begin_index:train_end_index],
                                    sos_id=SOS_TOKEN, eos_id=EOS_TOKEN,
                                    target_dict=target_dict,
                                    input_reverse=hparams.input_reverse,
                                    use_augment=hparams.use_augment,
                                    batch_size=hparams.batch_size,
                                    augment_ratio=hparams.augment_ratio,
                                    pack_by_length=hparams.pack_by_length
                                )
        )

    valid_dataset = BaseDataset(
                        audio_paths=audio_paths[train_num:],
                        label_paths=label_paths[train_num:],
                        sos_id=SOS_TOKEN, eos_id=EOS_TOKEN,
                        batch_size=hparams.batch_size,
                        target_dict=target_dict,
                        input_reverse=hparams.input_reverse,
                        use_augment=False,
                        pack_by_length=hparams.pack_by_length
    )

    logger.info("split dataset complete !!")
    return train_time_step, train_dataset_list, valid_dataset