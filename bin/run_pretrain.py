import argparse
import torch
import librosa
import sys
import numpy as np
sys.path.append('..')
from kospeech.vocab import KsponSpeechVocabulary
from torch import Tensor
from kospeech.data.audio.core import load_audio
from kospeech.model_builder import load_test_model


def parse_audio(audio_path: str, del_silence: bool = True) -> Tensor:
    signal = load_audio(audio_path, del_silence)
    mfcc = librosa.feature.mfcc(y=signal, sr=16000, n_mfcc=40, n_fft=320, hop_length=160)

    mfcc -= mfcc.mean()
    mfcc = Tensor(mfcc).transpose(0, 1)

    mfcc = mfcc[:, ::-1]
    mfcc = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(mfcc, 0, 1)))

    return mfcc


parser = argparse.ArgumentParser(description='Run Pretrain')
parser.add_argument('--model_path', type=str, default='../pretrain/model.pt')
parser.add_argument('--audio_path', type=str, default='../pretrain/sample_audio.pcm')
parser.add_argument('--device', type=str, default='cuda')
opt = parser.parse_args()

feature_vector = parse_audio(opt.audio_path, del_silence=True)
input_length = torch.IntTensor([len(feature_vector)])
vocab = KsponSpeechVocabulary('../data/vocab/aihub_vocabs.csv')

model = load_test_model(opt, opt.device)
model.eval()

output = model(inputs=feature_vector.unsqueeze(0), input_lengths=input_length,
               teacher_forcing_ratio=0.0, return_decode_dict=False)
logit = torch.stack(output, dim=1).to(opt.device)
pred = logit.max(-1)[1]

sentence = vocab.label_to_string(pred.cpu().detach().numpy())
print(sentence)
