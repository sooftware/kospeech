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

def load_data_list(data_list_path):
    """
    Provides set of audio path & label path
    Inputs: data_list_path
        data_list_path: csv file with training or test data list
    Outputs: audio_paths, label_paths
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label path
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
    """
    data_list = pd.read_csv(data_list_path, "r", delimiter = ",", encoding="UTF-8")
    audio_paths = list(DATASET_PATH + data_list["audio"])
    label_paths = list(DATASET_PATH + data_list["label"])
    return audio_paths, label_paths


def load_targets(label_paths):
    """
    Provides dictionary of filename and labels
    Inputs: label_paths
        - **label_paths**: set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
    Outputs: target_dict
        - **target_dict**: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    """
    target_dict = dict()
    for label_txt in label_paths:
        f = open(file=label_txt, mode="r")
        label = f.readline()
        f.close()
        file_num = label_txt.split('/')[-1].split('.')[0].split('_')[-1]
        target_dict['KaiSpeech_label_'+file_num] = label
    return target_dict


def get_label(label_path, bos_id=2037, eos_id=2038, target_dict=None):
    """
    Provides specific file`s label to list format.
    Inputs: filepath, bos_id, eos_id, target_dict
        - **filepath**: specific path of label file
        - **bos_id**: <s>`s id
        - **eos_id**: </s>`s id
        - **target_dict**: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    Outputs: label
        - **label**: list of bos + sequence of label + eos
                Format : [<s>, 5, 0, 49, 4, 0, 8, 190, 0, 78, 115, </s>]
    """
    if target_dict == None: logger.info("target_dict is None")
    key = label_path.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')

    label = list()
    label.append(bos_id)
    for token in tokens:
        label.append(int(token))
    label.append(eos_id)
    return label