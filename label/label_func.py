"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from definition import *

def load_label(label_path):
    """
    Provides char2index, index2char
    Inputs: label_path
        label_path: csv file with character labels
            Format : char | id | freq
    Outputs: char2index, index2char
        - **char2index**: char2index[ch] = id
        - **index2char**: index2char[id] = ch
    """
    char2index = dict()
    index2char = dict()
    f = open(label_path, 'r', encoding="UTF-8")
    labels = csv.reader(f, delimiter=',')
    header = next(labels)

    for row in labels:
        char2index[row[1]] = row[0]
        index2char[row[0]] = row[1]

    return char2index, index2char

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents