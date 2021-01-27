import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
import pdb
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kspeech.model_builder import build_model
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--device', type=str, required=False, default='cpu')
opt = parser.parse_args()

feature = parse_audio(opt.audio_path, del_silence=True)
input_length = torch.LongTensor([len(feature)])
vocab = KsponSpeechVocabulary('/home/sanghoon/KoSpeech/data/vocab/aihub_character_vocabs.csv')

pdb.set_trace()
model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
if isinstance(model, nn.DataParallel):
    model = model.module
model.eval()

if isinstance(model, ListenAttendSpell):
    model.encoder.device = opt.device
    model.decoder.device = opt.device

    y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
elif isinstance(model, DeepSpeech2):
    model.device = opt.device
    y_hats = model.greedy_search(feature.unsqueeze(0), input_length, opt.device)
elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
    y_hats = model.greedy_search(feature.unsqueeze(0).to(opt.device), input_length, opt.device)

sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
print(sentence)
