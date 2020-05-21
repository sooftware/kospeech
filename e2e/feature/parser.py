import numpy as np
import librosa
import torch
import platform
import random
from e2e.modules.definition import logger

if platform.system() == 'Linux':
    import torchaudio
    import from_librosa


class AudioParser(object):
    def parse_script(self, script_path):
        """
        Args:
            script_path: Path where script is stored from the manifest file

        Returns:
            Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        Args:
            audio_path: Path where audio is stored from the manifest file

        Returns:
            Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    """
    Parses audio file into spectrogram with various options.

    Args:
        feature_extract_by (str): which library to use for feature extraction: [librosa, torchaudio] (default: librosa)
        sample_rate (int): sample rate
        n_mels (int): number of mel filter
        window_size (int): window size (ms)
        stride (int): forwarding size (ms)
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)
        normalize (bool): flag indication whether to normalize spectrum or not (default:True)
        time_mask_para (int): Hyper Parameter for Time Masking to limit time masking length
        freq_mask_para (int): Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
    """

    def __init__(self, feature_extract_by='librosa', sample_rate=16000, n_mels=80, window_size=20, stride=10,
                 del_silence=False, input_reverse=True, normalize=False,
                 time_mask_para=70, freq_mask_para=12, time_mask_num=2, freq_mask_num=2,
                 sos_id=1, eos_id=2, target_dict=None):
        super(SpectrogramParser, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.n_fft = int(sample_rate * 0.001 * window_size)
        self.hop_length = int(sample_rate * 0.001 * stride)
        self.time_mask_para = time_mask_para
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
        self.feature_extract_by = feature_extract_by
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.target_dict = target_dict

        if feature_extract_by == 'torchaudio':
            self.transforms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,  win_length=window_size,
                                                                   hop_length=self.hop_length,  n_fft=self.n_fft,
                                                                   n_mels=n_mels)
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def load_audio(self, audio_path):
        if audio_path.endswith('.pcm'):
            try:
                pcm = np.memmap(audio_path, dtype='h', mode='r')
            except RuntimeError:
                logger.info('RuntimeError in {0}'.format(audio_path))
                return None

            signal = np.array([float(x) for x in pcm])

        elif audio_path.endswith('.wav'):
            signal, _ = librosa.core.load(audio_path, sr=self.sample_rate)

        else:
            raise ValueError("Unsupported format: {0}".format(audio_path.split('.')[-1]))

        return signal

    def parse_audio(self, audio_path):
        signal = self.load_audio(audio_path)

        if signal is None:  # Exception handling
            return None

        if self.del_silence:
            non_silence_ids = from_librosa.split(y=signal, top_db=30)
            signal = np.concatenate([signal[start:end] for start, end in non_silence_ids])

        if self.feature_extract_by == 'torchaudio':
            spectrogram = self.transforms(torch.FloatTensor(signal))
            spectrogram = self.amplitude_to_db(spectrogram)
            spectrogram = spectrogram.numpy()

        else:
            spectrogram = librosa.feature.melspectrogram(signal, self.sample_rate, n_mels=self.n_mels,
                                                         n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        if self.normalize:
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram -= mean
            spectrogram /= std

        if self.input_reverse:
            spectrogram = spectrogram[:, ::-1]

        spectrogram = torch.FloatTensor(np.ascontiguousarray(np.swapaxes(spectrogram, 0, 1)))
        return spectrogram

    def parse_script(self, script_path):
        scripts = list()

        key = script_path.split('/')[-1].split('.')[0]
        script = self.target_dict[key]

        tokens = script.split(' ')

        scripts.append(int(self.sos_id))

        for token in tokens:
            scripts.append(int(token))

        scripts.append(int(self.eos_id))
        return scripts

    def spec_augment(self, spectrogram):
        """ Provides SpecAugmentation for audio """
        length = spectrogram.size(0)
        n_mels = spectrogram.size(1)

        # time mask
        for _ in range(self.time_mask_num):
            t = np.random.uniform(low=0.0, high=self.time_mask_para)
            t = int(t)
            if length - t > 0:
                t0 = random.randint(0, length - t)
                spectrogram[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = np.random.uniform(low=0.0, high=self.freq_mask_para)
            f = int(f)
            f0 = random.randint(0, n_mels - f)
            spectrogram[:, f0: f0 + f] = 0

        return spectrogram
