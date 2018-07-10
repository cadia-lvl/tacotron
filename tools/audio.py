import librosa
import numpy as np
import tensorflow as tf
from scipy import signal

from hparams import hparams

def load_wav(path):
    '''
        Loads a single waveform file from
        disk at the given path
    '''
    return librosa.core.load(path, sr=hparams.get('sample_rate'))[0]

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.float16), hparams.get('sample_rate'))

def trim_silence(audio):
    trimmed, _ = librosa.effects.trim(audio)
    return trimmed

def spectrogram(y):
    '''
        Input
        y: a numpy array representing a sound signal
            
        Output
        A normalized linear-scale spectrogram. A spectrogram is 
        a 2d structure ([Time (ms), Frequency (Hz)] where values
        are  Volume (dB))
        TODO Thresholding at ref_level_db is never discussed in
        the tacotron paper
    '''
    # D is the short-time Fourier transform result of
    # the pre-emphasizes version of the input signal
    D = _stft(pre_emphasis(y))
    # Convert to a dB-scaled spectrogram and threshold
    # the output at ref_level_db
    S = _amp_to_db(np.abs(D)) - hparams.get('ref_level_db')
    # Finally normalize the output
    return _normalize(S)

def spectrogram_inv(spect):
    '''
        Input
        spect: A linear spectrogram

        Convert a spectrogram back to a waveform using the
        Griffin-lim algorithm. This is used in synthesizing
    '''
    # Unwind normalization and dB-scaling
    S = _db_to_amp(_normalize_inv(spect) + hparams.get('ref_level_db'))
    # Apply the Griffin-lim algorithm and unwind the pre-emphasis
    return pre_emphasis_inv(_griffin_lim(S ** hparams.get('power')))

def spectrogram_tensorflow_inv(spect):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

    Unlike spectrogram_inv, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
  '''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spect) + hparams.get('ref_level_db'))
  return _griffin_lim_tensorflow(tf.pow(S, hparams.get('power')))

def mel_spectrogram(y):
    '''
        Input
        y: a numpy array representing a sound signal

        Output
        A normalized mel-scaled spectrogram. A spectrogram is 
        a 3d structure (Time (ms), Frequency (Hz), Volume (dB))
        TODO Thresholding at ref_level_db is never discussed in
        the tacotron paper
    '''
    D = _stft(pre_emphasis(y))
    S = _amp_to_db(_linspect_to_melspect(np.abs(D))) - hparams.get('ref_level_db')
    return _normalize(S)

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(hparams.get('sample_rate') * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
            return x + hop_length
    return len(wav)


def pre_emphasis(x):
    '''
        Input
        x: a numpy array representing a sound signal

        Output
        Applies a pre-emphasis filter on the signal to amplify
        the high frequencies. Given an input signal x, the emphasized
        signal y is described by

            y(t) = x(t) - a*x(t-1),

        where a is the pre emphasis coefficient. 
        
        This is done with lfilter where lfilter(a, b, x) implements
        a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                  - a[1]*y[n-1] - ... - a[N]*y[n-N] 
    '''
    return signal.lfilter([1, -float(hparams.get('preemphasis'))], [1], x)

def pre_emphasis_inv(x):
    '''
        Rewinds the pre emphasis filter. This is used
        in synthesizing
    '''
    return signal.lfilter([1], [1, -float(hparams.get('preemphasis'))], x)


def _stft(y):
    '''
        Input
        x: a numpy array representing a sound signal

        Output
        Applies the librosa Short-time Fourier transform, givegn
        the hyperparameters. It returns a complex-valued matrix D
        such that:
            * np.abs(D[f,t]) is the magnitude of frequency bin f at frame t
            * np.angle(D[f,t]) is the phase of frequency bin f at frame t
        This returns an amplitude-scaled Spectrogram (not dB-scaled)
    '''
    n_fft, hop_length, win_length = _stft_params()
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, 
        win_length=win_length, window='hann')

def _stft_inv(spect):
  '''
    Input
    spect: Spectrogram

    Returns the inverse short-time Fourier transform (ISTFT).
    Converts a complex-valued spectrogram stft_matrix to 
    time-series y by minimizing the mean squared error between 
    spect and STFT of y._
  '''
  _, hop_length, win_length = _stft_params()
  return librosa.istft(spect, hop_length=hop_length, 
        win_length=win_length, window='hann')

def _stft_params():
    '''
        Output
        Given the hyper parameters, return the needed
        parameters for the lirosa STFT method
    
        n_fft: The FFT window size or the num
        hop_length: The number of audio frames between STFT columns
        win_length: Each frame of audio is windowed, where each window
        will be of length win_length and then zero-padded to match up with n_fft
    '''
    n_fft = hparams.get('n_fft')
    hop_length = int(hparams.get('frame_shift_ms') / 1000 * hparams.get('sample_rate'))
    win_length = int(hparams.get('frame_length_ms') / 1000 * hparams.get('sample_rate'))
    return n_fft, hop_length, win_length


def _griffin_lim(spect):
    '''
        Input
        spect: A spectrogram

        Apply the Griffin-Lim Algorithm (GLA) on the spectrogram
        to estimate the signal that has been STFTed
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*spect.shape))
    S_complex = np.abs(spect).astype(np.complex)
    y = _stft_inv(S_complex * angles)
    for _ in range(hparams.get('griffin_lim_iters')):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _stft_inv(S_complex * angles)
    return y

def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.get('griffin_lim_iters')):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_params()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_params()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


# Conversions

_mel_basis = None

def _amp_to_db(x):
    '''
        Converts a amplitude-scaled spectrogram to a
        dB-scaled spectrogram
    '''
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    '''
        Converts a dB-scaled spectrogram to a amplitude-scaled
        spectrogram
    '''
    return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _linspect_to_melspect(spect):
    '''
        input
        spect: A linear spectrogram

        Transforms a linear spectrogram into a mel
        spectrogram. A mel spectrogram's frequencey bands
        are equally spaced on the mel scale which allows for
        a better representation of sound.

        f (hertz) -> m (mels): m = 2595 log_10(1+f/700)
    '''
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spect)

def _build_mel_basis():
    '''
        Creates a filterbank matrix to combine FFT bins into
        mel-frequency bins
    '''
    return librosa.filters.mel(hparams.get('sample_rate'), 
        hparams.get('n_fft'), n_mels=hparams.get('num_mels'))

def _normalize(S):
    '''
        Input
        S: Spectrogram

        Returns a normalized version of the spectrogram.
        Since we don't care about absolute volume and only
        care about relatve volume, we pin the spectrogram frequency

    '''
    return np.clip((S - hparams.get('min_level_db')) / - float(hparams.get('min_level_db')), 0, 1)

def _normalize_inv(S):
    '''
        Input
        S: Spectrogram

        Unwinds the normalization function applied
        to the spectrogram. This is used in synthesizing
    '''
    return (np.clip(S, 0, 1) * -float(hparams.get('min_level_db'))) + hparams.get('min_level_db')

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -float(hparams.get('min_level_db'))) + hparams.get('min_level_db')
