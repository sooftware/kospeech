import torch
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from e2e.data.preprocess.core import split

MODEL_PATH = ''
AUDIO_PATH = ''
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

    spectrogram = librosa.feature.melspectrogram(sound, 16000, n_mels=80, n_fft=320, hop_length=160)
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
model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

_, alignment = model(spectrogram.unsqueeze(0), torch.IntTensor([len(spectrogram)]))
alignment = alignment.squeeze(0)

alignment = pd.DataFrame(alignment)

f = plt.figure(figsize=(16, 6))
plt.imshow(spectrogram.transpose(0, 1), aspect='auto', origin='lower')

g = plt.figure(figsize=(10, 8))
sns.heatmap(alignment, fmt="f", cmap='viridis')

f.show()
g.show()
