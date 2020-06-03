import torch
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from e2e.data.preprocess.core import split

MODEL_PATH = '../data/checkpoints/model.pt'
AUDIO_PATH = '../data/sample/KaiSpeech_000002.pcm'
DEL_SILENCE = True
NORMALIZE = True
FEATURE_EXTRACT_BY = 'librosa'
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

_, alignments = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]), teacher_forcing_ratio=0.0)  # D(NxT)
attention_map = None

for decode_timestep in alignments:
    if attention_map is None:
        attention_map = decode_timestep
    else:
        attention_map = torch.cat([attention_map, decode_timestep], dim=1)  # NxDxT

attention_map = torch.flip(attention_map, dims=[0, 1])
num_heads = attention_map.size(0)

f = plt.figure(figsize=(16, 6))
plt.imshow(spectrogram.transpose(0, 1), aspect='auto', origin='lower')
plt.savefig("./image/spectrogram.png")

for n in range(num_heads):
    g = plt.figure(figsize=(10, 8))
    map = pd.DataFrame(attention_map[n].cpu().detach().numpy().transpose(0, 1))
    map = sns.heatmap(map, fmt="f", cmap='viridis')
    map.invert_yaxis()
    fig = map.get_figure()
    fig.savefig("./image/head%s.png" % str(n))
