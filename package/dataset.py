import math
import random
from torch.utils.data import Dataset
from package.definition import EOS_token, logger, SOS_token
from package.feature import spec_augment, get_librosa_melspectrogram
from package.utils import get_label, save_pickle


class CustomDataset(Dataset):
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
        batch_size (int): mini batch size
    """

    def __init__(self, audio_paths, label_paths, sos_id, eos_id,
                 target_dict=None, input_reverse=True, use_augment=True,
                 batch_size=None, augment_ratio=1.0):
        self.audio_paths = list(audio_paths)
        self.label_paths = list(label_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_size = batch_size
        self.target_dict = target_dict
        self.input_reverse = input_reverse
        self.augment_ratio = augment_ratio
        self.augment_flags = [False] * len(self.audio_paths)

        if use_augment:
            self.augmentation()

        bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
        random.shuffle(bundle)
        self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def get_item(self, idx):
        label = get_label(self.label_paths[idx], sos_id=self.sos_id, eos_id=self.eos_id, target_dict=self.target_dict)
        feat = get_librosa_melspectrogram(self.audio_paths[idx], n_mels=80, input_reverse=self.input_reverse)

        if feat is None:  # exception handling
            return None, None

        if self.augment_flags[idx]:
            feat = spec_augment(feat, T=70, F=15, time_mask_num=2, freq_mask_num=2)

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
        bundle = list(zip(self.audio_paths, self.label_paths, self.augment_flags))
        random.shuffle(bundle)
        self.audio_paths, self.label_paths, self.augment_flags = zip(*bundle)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


def split_dataset(config, audio_paths, label_paths, valid_ratio=0.05, target_dict=None):
    """
    Dataset split into training and validation Dataset.

    Args:
        valid_ratio: validation set ratio of total dataset
        config (package.config.HyperParams): set of configures
        audio_paths (list): set of audio path
        label_paths (list): set of label path
        target_dict (dict): dictionary of filename and target

    Returns: train_batch_num, train_dataset_list, valid_dataset
        - **train_batch_num** (int): num of batch for training
        - **train_dataset_list** (list): list of training dataset
        - **valid_dataset** (utils.dataset.BaseDataset): validation dataset
    """
    logger.info("split dataset start !!")

    trainset_list = list()
    train_num = math.ceil(len(audio_paths) * (1 - valid_ratio))
    total_time_step = math.ceil(len(audio_paths) / config.batch_size)
    valid_time_step = math.ceil(total_time_step * valid_ratio)
    train_time_step = total_time_step - valid_time_step

    if config.use_augment:
        train_time_step = int(train_time_step * (1 + config.augment_ratio))

    train_num_per_worker = math.ceil(train_num / config.worker_num)

    # audio_paths & label_paths shuffled in the same order
    # for seperating train & validation
    data_paths = list(zip(audio_paths, label_paths))
    random.shuffle(data_paths)
    audio_paths, label_paths = zip(*data_paths)

    # seperating the train dataset by the number of workers
    for idx in range(config.worker_num):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)

        trainset_list.append(CustomDataset(
            audio_paths=audio_paths[train_begin_idx:train_end_idx],
            label_paths=label_paths[train_begin_idx:train_end_idx],
            sos_id=SOS_token, eos_id=EOS_token,
            target_dict=target_dict,
            input_reverse=config.input_reverse,
            use_augment=config.use_augment,
            batch_size=config.batch_size,
            augment_ratio=config.augment_ratio
        ))

    validset = CustomDataset(
        audio_paths=audio_paths[train_num:],
        label_paths=label_paths[train_num:],
        sos_id=SOS_token, eos_id=EOS_token,
        batch_size=config.batch_size,
        target_dict=target_dict,
        input_reverse=config.input_reverse,
        use_augment=False
    )

    save_pickle(trainset_list, './data/pickle/trainset_list')
    save_pickle(validset, './data/pickle/validset')

    logger.info("split dataset complete !!")

    return train_time_step, trainset_list, validset
