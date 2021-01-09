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

import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import torchaudio
sys.path.append('..')
from torch import Tensor
from kospeech.models.deepspeech2.model import DeepSpeech2
from kospeech.models.las.model import ListenAttendSpell
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature_vector = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature_vector -= feature_vector.mean()
    feature_vector /= np.std(feature_vector)

    return torch.FloatTensor(feature_vector).transpose(0, 1)


parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, require=True)
parser.add_argument('--audio_path', type=str, require=True)
parser.add_argument('--device', type=str, require=False, default='cpu')
opt = parser.parse_args()

feature_vector = parse_audio(opt.audio_path, del_silence=True)
input_length = torch.IntTensor([len(feature_vector)])
vocab = KsponSpeechVocabulary('data/vocab/aihub_character_vocabs.csv')

model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
if isinstance(model, nn.DataParallel):
    model = model.module
model.eval()

if isinstance(model, ListenAttendSpell):
    model.encoder.device = opt.device
    model.decoder.device = opt.device

    y_hats = model.greedy_search(feature_vector.unsqueeze(0), input_length, opt.device)
elif isinstance(model, DeepSpeech2):
    model.device = opt.device
    y_hats = model.greedy_search(feature_vector.unsqueeze(0), input_length, opt.device)

sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
print(sentence)
