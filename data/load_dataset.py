import math
import pickle

def load_dataset(hparams, audio_paths, valid_ratio=0.05):
    batch_num = math.ceil(len(audio_paths) / hparams.batch_size)
    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num
    if hparams.use_augment: train_batch_num *= int(1 + hparams.augment_ratio)

    with open('./pickle/train_dataset.txt', 'rb') as f:
        train_dataset = pickle.load(f)

    with open('./pickle/valid_dataset.txt', 'rb') as f:
        valid_dataset = pickle.load(f)

    return train_batch_num, train_dataset, valid_dataset