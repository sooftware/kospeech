# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
from kospeech.vocabs import Vocabulary


class KsponSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, output_unit: str = 'character', sp_model_path=None):
        super(KsponSpeechVocabulary, self).__init__()
        if output_unit == 'subword':
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sp_model_path)

            self.pad_id = 0
            self.sos_id = 1
            self.eos_id = 2
            self.blank_id = len(self)
        else:
            self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding='utf-8')
            self.sos_id = int(self.vocab_dict['<sos>'])
            self.eos_id = int(self.vocab_dict['<eos>'])
            self.pad_id = int(self.vocab_dict['<pad>'])
            self.blank_id = int(self.vocab_dict['<blank>'])
            self.labels = self.vocab_dict.keys()

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
                return self.sp.DecodeIds([int(l) for l in labels])

            sentences = list()
            for batch in labels:
                sentence = str()
                for label in batch:
                    sentence = self.sp.DecodeIds([int(l) for l in label])
                sentences.append(sentence)
            return sentences

        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                  continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
        """
        Provides char2id, id2char

        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method

        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
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

                unit2id['<blank>'] = len(unit2id)
                id2unit[len(unit2id)] = '<blank>'

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))