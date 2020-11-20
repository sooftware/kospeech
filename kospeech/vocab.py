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
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


class KsponSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, output_unit: str = 'character', sp_model_path=None):
        if output_unit == 'subword':
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sp_model_path)
        else:
            self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding='utf-8')
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])

        self.vocab_path = vocab_path
        self.output_unit = output_unit

    def __len__(self):
        if self.output_unit == 'subword':
            count = 0
            with open(self.vocab_path, encoding='utf-8') as f:
                for _ in f.readlines():
                    count += 1

            return count
        return len(self.vocab_dict)

    def label_to_string(self, labels):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if self.output_unit == 'subword':
            if len(labels.shape) == 1:
                return self.sp.DecodeIds([l for l in labels])

            sentences = list()
            for batch in labels:
                sentence = str()
                for label in batch:
                    sentence = self.sp.DecodeIds([l for l in label])
                sentences.append(sentence)
            return sentences

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
        unit2id = dict()
        id2unit = dict()

        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))


class LibriSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, model_path):
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

        self.vocab_path = vocab_path

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def __len__(self):
        count = 0
        with open(self.vocab_path, encoding='utf-8') as f:
            for _ in f.readlines():
                count += 1

        return count

    def label_to_string(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([l for l in labels])

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                sentence = self.sp.DecodeIds([l for l in label])
            sentences.append(sentence)
        return sentences