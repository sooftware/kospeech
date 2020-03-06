"""
Copyright 2020- Kai.Lib

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def get_label(filepath, sos_id=2037, eos_id=2038, target_dict=None):
    """
    Provides specific file`s label to list format.

    Parameters
    -----------
        - **filepath** (str): specific path of label file
        - **bos_id** (int): identification of <start of sequence>
        - **eos_id** (int): identification of <end of sequence>
        - **target_dict** (dict): dictionary of filename and labels

    Returns
    --------
        - **label** (list): list of bos + sequence of label + eos
    """
    assert target_dict is not None, "target_dict is None"
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')

    label = list()
    label.append(int(sos_id))
    for token in tokens:
        label.append(int(token))
    label.append(int(eos_id))
    return label

def label_to_string(labels, id2char, eos_id):
    """
    Converts label to string (number => Hangeul)

    Parameters
    -----------
        - **labels**: number label
        - **id2char**: id2char[id] = ch
        - **eos_id**: identification of <end of sequence>

    Returns
    --------
        - **sentence** (str or list): Hangeul representation of labels
    """
    if len(labels.shape) == 1:
        sentence = str()
        for label in labels:
            if label.item() == eos_id:
                break
            sentence += id2char[label.item()]
        return sentence

    elif len(labels.shape) == 2:
        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == eos_id:
                    break
                sentence += id2char[label.item()]
            sentences.append(sentence)
        return sentences