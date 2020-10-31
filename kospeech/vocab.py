# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import csv
import sentencepiece as spm


class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def label_to_string(self, labels):
        raise NotImplementedError


class KsponSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path):
        self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding='utf-8')
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])
        self.vocab_size = len(self.vocab_dict)

    def label_to_string(self, labels):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
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


class LibriSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, model_path):
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

        with open(vocab_path, encoding='utf-8') as f:
            count = 0
            for _ in f.readlines():
                count += 1
            self.vocab_size = count

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def label_to_string(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIdx([l for l in labels])

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                sentence = self.sp.DecodeIdx([l for l in label])
            sentences.append(sentence)
        return sentences