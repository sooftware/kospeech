import warnings
import numpy as np
from astropy.modeling import ParameterError
from numpy.lib.stride_tricks import as_strided
from kospeech.utils import logger


def load_audio(audio_path, del_silence):
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    try:
        sound = np.memmap(audio_path, dtype='h', mode='r').astype('float32')

        if del_silence:
            non_silence_indices = split(sound, top_db=30)
            sound = np.concatenate([sound[start:end] for start, end in non_silence_indices])

        return sound / 32767  # normalize audio

    except ValueError:
        logger.debug('ValueError in {0}'.format(audio_path))
        return None
    except RuntimeError:
        logger.debug('RuntimeError in {0}'.format(audio_path))
        return None
    except IOError:
        logger.debug('IOError in {0}'.format(audio_path))
        return None


def __power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def __to_mono(y):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    def valid_audio(y, mono=True):
        if not isinstance(y, np.ndarray):
            raise ParameterError('Audio data must be of type numpy.ndarray')

        if not np.issubdtype(y.dtype, np.floating):
            raise ParameterError('Audio data must be floating-point')

        if mono and y.ndim != 1:
            raise ParameterError('Invalid shape for monophonic audio: '
                                 'ndim={:d}, shape={}'.format(y.ndim, y.shape))

        elif y.ndim > 2 or y.ndim == 0:
            raise ParameterError('Audio data must have shape (samples,) or (channels, samples). '
                                 'Received shape={}'.format(y.shape))

        if not np.isfinite(y).all():
            raise ParameterError('Audio buffer is not finite everywhere')

        if not y.flags["F_CONTIGUOUS"]:
            raise ParameterError('Audio buffer is not Fortran-contiguous. '
                                 'Use numpy.asfortranarray to ensure Fortran contiguity.')

        return True

    # Ensure Fortran contiguity.
    y = np.asfortranarray(y)

    # Validate the buffer.  Stereo is ok here.
    valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


def __frame(x, frame_length=2048, hop_length=512, axis=-1):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    if not isinstance(x, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(x)={}'.format(type(x)))

    if x.shape[axis] < frame_length:
        raise ParameterError('Input is too short (n={:d})'
                             ' for frame_length={:d}'.format(x.shape[axis], frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        if not x.flags['F_CONTIGUOUS']:
            raise ParameterError('Input array must be F-contiguous '
                                 'for framing along axis={}'.format(axis))

        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        if not x.flags['C_CONTIGUOUS']:
            raise ParameterError('Input array must be C-contiguous '
                                 'for framing along axis={}'.format(axis))

        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)
    else:
        raise ParameterError('Frame axis={} must be either 0 or -1'.format(axis))

    return as_strided(x, shape=shape, strides=strides)


def __rms(y=None, S=None, frame_length=2048, hop_length=512,
          center=True, pad_mode='reflect'):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    if y is not None:
        y = __to_mono(y)
        if center:
            y = np.pad(y, int(frame_length // 2), mode=pad_mode)

        x = __frame(y,
                    frame_length=frame_length,
                    hop_length=hop_length)

        # Calculate power
        power = np.mean(np.abs(x) ** 2, axis=0, keepdims=True)
    elif S is not None:
        # Check the frame length
        if S.shape[0] != frame_length // 2 + 1:
            raise ParameterError(
                'Since S.shape[0] is {}, '
                'frame_length is expected to be {} or {}; '
                'found {}'.format(
                    S.shape[0],
                    S.shape[0] * 2 - 2, S.shape[0] * 2 - 1,
                    frame_length))

        # power spectrogram
        x = np.abs(S) ** 2

        # Adjust the DC and sr/2 component
        x[0] *= 0.5
        if frame_length % 2 == 0:
            x[-1] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=0, keepdims=True) / frame_length ** 2
    else:
        raise ParameterError('Either `y` or `S` must be input.')

    return np.sqrt(power)


def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60,
                               ref=np.max):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    # Convert to mono
    y_mono = __to_mono(y)

    # Compute the MSE for the signal
    mse = __rms(y=y_mono,
                frame_length=frame_length,
                hop_length=hop_length) ** 2

    return __power_to_db(mse.squeeze(), ref=ref, top_db=None) > - top_db


def _frames_to_samples(frames, hop_length=512, n_fft=None):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def split(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    """
    codes from https://github.com/librosa/librosa
    use this code fragments instead of importing librosa package,
    because of our server has a problem with importing librosa.
    """
    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)

    # Interval slicing, adapted from
    # https://stackoverflow.com/questions/2619413/efficiently-finding-the-interval-with-non-zeros-in-scipy-numpy-in-python
    # Find points where the sign flips
    edges = np.flatnonzero(np.diff(non_silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had high energy, count it
    if non_silent[0]:
        edges.insert(0, [0])

    # Likewise for the last frame
    if non_silent[-1]:
        edges.append([len(non_silent)])

    # Convert from frames to samples
    edges = _frames_to_samples(np.concatenate(edges),
                               hop_length=hop_length)

    # Clip to the signal duration
    edges = np.minimum(edges, y.shape[-1])

    # Stack the results back as an ndarray
    return edges.reshape((-1, 2))
