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

import torch
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kospeech.data.audio.core import split
from kospeech.models.seq2seq.decoder import Seq2seqDecoder

MODEL_PATH = '../data/checkpoints/model.pt'
AUDIO_PATH = '../data/sample/KaiSpeech_000098.pcm'
DEL_SILENCE = True
NORMALIZE = True
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 320
HOP_LENGTH = 160


def load_audio(audio_path, del_silence):
    sound = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

    if del_silence:
        non_silence_indices = split(sound, top_db=30)
        sound = np.concatenate([sound[start:end] for start, end in non_silence_indices])

    sound /= 32767  # normalize audio
    return sound


# Set your parse_audio() method used in training.
def parse_audio(audio_path):
    sound = load_audio(audio_path, DEL_SILENCE)

    spectrogram = librosa.feature.melspectrogram(sound, SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    if NORMALIZE:
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram -= mean
        spectrogram /= std

    spectrogram = spectrogram[:, ::-1]
    spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))

    return spectrogram


spectrogram = parse_audio(AUDIO_PATH)
model = torch.load(MODEL_PATH)

_, metadata = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]), teacher_forcing_ratio=0.0)  # D(NxT)

alignments = metadata[Seq2seqDecoder.KEY_ATTN_SCORE]
attention_maps = None

for decode_timestep in alignments:
    if attention_maps is None:
        attention_maps = decode_timestep
    else:
        attention_maps = torch.cat([attention_maps, decode_timestep], dim=1)  # NxDxT

attention_maps = torch.flip(attention_maps, dims=[0, 1])
num_heads = attention_maps.size(0)

f = plt.figure(figsize=(16, 6))
plt.imshow(spectrogram.transpose(0, 1), aspect='auto', origin='lower')
plt.savefig("./image/spectrogram.png")

for n in range(num_heads):
    g = plt.figure(figsize=(10, 8))
    attention_map = pd.DataFrame(attention_maps[n].cpu().detach().numpy().transpose(0, 1))
    attention_map = sns.heatmap(attention_map, fmt="f", cmap='viridis')
    attention_map.invert_yaxis()
    fig = attention_map.get_figure()
    fig.savefig("./image/head%s.png" % str(n))
