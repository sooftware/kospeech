# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import csv
from tqdm import trange


def load_vocab(label_path, encoding='utf-8'):
    """
    Provides char2id, id2char

    Args:
        label_path (str): csv file with character labels
        encoding (str): encoding method

    Returns: char2id, id2char
        - **char2id** (dict): char2id[ch] = id
        - **id2char** (dict): id2char[id] = ch
    """
    char2id = dict()
    id2char = dict()

    try:
        with open(label_path, 'r', encoding=encoding) as f:
            labels = csv.reader(f, delimiter=',')
            next(labels)

            for row in labels:
                char2id[row[1]] = row[0]
                id2char[int(row[0])] = row[1]

        return char2id, id2char
    except IOError:
        raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))


def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels

    Args:
        transcripts_path (str): path of transcripts

    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    audio_paths = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, korean_transcript, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts
