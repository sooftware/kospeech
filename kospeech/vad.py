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
import librosa
import numpy as np
from kospeech.models import ResnetVADModel


class VoiceActivityDetection(object):
    """
    Voice activity detection (VAD), also known as speech activity detection or speech detection, 
    is the detection of the presence or absence of human speech, used in speech processing.

    Please use our pre-train model.
    Refer to : https://github.com/sooftware/KoSpeech

    Args: model_path, device
        model_path: path of vad model 
        device: 'cuda' or 'cpu'
    """

    def __init__(self, model_path: str, device: str):
        self.sample_rate = 16000
        self.n_mfcc = 5
        self.n_mels = 40
        self.device = device

        self.model = ResnetVADModel()
        self.model.load_state_dict(torch.load(model_path))

        self.model.to(device)
        self.model.eval()

    def extract_features(
            self,
            signal,
            size: int = 512,
            step: int = 16,
    ):
        # Mel Frequency Cepstral Coefficents
        mfcc = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=size, hop_length=step)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Root Mean Square Energy
        melspectrogram = librosa.feature.melspectrogram(y=signal, n_mels=self.n_mels, sr=self.sample_rate,
                                                        n_fft=size, hop_length=step)
        rmse = librosa.feature.rms(S=melspectrogram, frame_length=self.n_mels * 2 - 1, hop_length=step)

        mfcc = np.asarray(mfcc)
        mfcc_delta = np.asarray(mfcc_delta)
        mfcc_delta2 = np.asarray(mfcc_delta2)
        rmse = np.asarray(rmse)

        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)
        features = np.transpose(features)

        return features

    def smooth_predictions_v1(self, label):
        smoothed_label = list()

        # Smooth with 3 consecutive windows
        for i in range(2, len(label), 3):
            cur_pred = label[i]
            if cur_pred == label[i - 1] == label[i - 2]:
                smoothed_label.extend([cur_pred, cur_pred, cur_pred])
            else:
                if len(smoothed_label) > 0:
                    smoothed_label.extend([smoothed_label[-1], smoothed_label[-1], smoothed_label[-1]])
                else:
                    smoothed_label.extend([0, 0, 0])

        n = 0
        while n < len(smoothed_label):
            cur_pred = smoothed_label[n]
            if cur_pred == 1:
                if n > 0:
                    smoothed_label[n - 1] = 1
                if n < len(smoothed_label) - 1:
                    smoothed_label[n + 1] = 1
                n += 2
            else:
                n += 1

        for idx in range(len(label) - len(smoothed_label)):
            smoothed_label.append(smoothed_label[-1])

        return smoothed_label

    def smooth_predictions_v2(self, label):
        smoothed_label = list()
        # Smooth with 3 consecutive windows
        for i in range(2, len(label)):
            cur_pred = label[i]
            if cur_pred == label[i - 1] == label[i - 2]:
                smoothed_label.append(cur_pred)
            else:
                if len(smoothed_label) > 0:
                    smoothed_label.append(smoothed_label[-1])
                else:
                    smoothed_label.append(0)

        n = 0
        while n < len(smoothed_label):
            cur_pred = smoothed_label[n]
            if cur_pred == 1:
                if n > 0:
                    smoothed_label[n - 1] = 1
                if n < len(smoothed_label) - 1:
                    smoothed_label[n + 1] = 1
                n += 2
            else:
                n += 1

        for idx in range(len(label) - len(smoothed_label)):
            smoothed_label.append(smoothed_label[-1])

        return smoothed_label

    def get_speech_intervals(self, data, label):
        def get_speech_interval(labels):
            seguence_length = 1024
            speech_interval = [[0, 0]]
            pre_label = 0

            for idx, label in enumerate(labels):

                if label:
                    if pre_label == 1:
                        speech_interval[-1][1] = (idx + 1) * seguence_length
                    else:
                        speech_interval.append([idx * seguence_length, (idx + 1) * seguence_length])

                pre_label = label

            return speech_interval[1:]

        speech_intervals = list()
        interval = get_speech_interval(label)

        for start, end in interval:
            speech_intervals.append(data[start:end])

        return speech_intervals

    def __call__(self, audio_path: str = None, sample_rate: int = 16000):
        seguence_signal = list()
        self.sample_rate = sample_rate

        start_pointer = 0
        end_pointer = 1024

        if audio_path.endswith('.wav') or audio_path.endswith('.flac'):
            signal, _ = librosa.load(audio_path, sr=self.sample_rate)

        elif audio_path.endswith('.pcm'):
            signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

        else:
            raise ValueError(f"Unsupported Format : {audio_path}")

        while end_pointer < len(signal):
            seguence_signal.append(signal[start_pointer:end_pointer])

            start_pointer = end_pointer
            end_pointer += 1024

        feature = [self.extract_features(signal) for signal in seguence_signal]

        feature = np.array(feature)
        feature = np.expand_dims(feature, 1)
        x_tensor = torch.from_numpy(feature).float().to(self.device)

        output = self.model(x_tensor)
        predicted = torch.max(output.data, 1)[1]

        predict_label = predicted.to(torch.device('cpu')).detach().numpy()

        predict_label = self.smooth_predictions_v2(predict_label)
        predict_label = self.smooth_predictions_v1(predict_label)

        return self.get_speech_intervals(signal, predict_label)
