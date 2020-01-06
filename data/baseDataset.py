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

from torch.utils.data import Dataset
from feature.feature import get_librosa_melspectrogram
from loader.loader import get_label

class BaseDataset(Dataset):
    """
    Inputs: audio_paths, label_paths, bos_id, eos_id, target_dict
        - **audio_paths**: set of audio path
                Format : [base_dir/KaiSpeech/KaiSpeech_123260.pcm, ... , base_dir/KaiSpeech/KaiSpeech_621245.pcm]
        - **label_paths**: set of label paths
                Format : [base_dir/KaiSpeech/KaiSpeech_label_123260.txt, ... , base_dir/KaiSpeech/KaiSpeech_label_621245.txt]
        - **bos_id**: <s>`s id
        - **eos_id**: </s>`s id
        - **target_dict**: dictionary of filename and labels
                Format : {KaiSpeech_label_FileNum : '5 0 49 4 0 8 190 0 78 115', ... }
    """
    def __init__(self, audio_paths, label_paths, bos_id = 2037, eos_id = 2038, target_dict = None):
        self.audio_paths = audio_paths
        self.label_paths = label_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.target_dict = target_dict

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

    def getitem(self, idx):
        # 리스트 형식으로 label을 저장
        script = get_label(self.label_paths[idx], self.bos_id, self.eos_id, self.target_dict)
        # 음성데이터에 대한 feature를 feat에 저장 -> tensor 형식
        feat = get_librosa_melspectrogram(self.audio_paths[idx], n_mels = 80, del_silence = True, mel_type='log_mel')
        return feat, script