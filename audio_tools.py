from scipy import signal
from hparams import hparams
import librosa
import numpy as np


def spectogram(y):
    '''
        Input
        y: a numpy array representing a sound signal
            
        Output
        A normalized linear-scale spectrogram. A spectrogram is 
        a 3d structure (Time (ms), Frequency (Hz), Volume (dB))
        TODO Thresholding at ref_level_db is never discussed in
        the tacotron paper
    '''
    # D is the short-time Fourier transform result of
    # the pre-emphasizes version of the input signal
    D = _stft(pre_emphasis(y))
    # Convert to a dB-scaled spectrogram and threshold
    # the output at ref_level_db
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    # Finally normalize the output
    return _normalize(S)

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
    S = _amp_to_db(_linspect_to_melspect(np.abs(D))) - hparams.ref_level_db
    return _normalize(S)

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
    return signal.lfilter([1, -hparams.preemphasis], [1], x)


def _stft(y):
    '''
        Input
        x: a numpy array representing a sound signal

        Output
        Applies the librosa Short-time Fourier transform, given
        the hyperparameters. It returns a complex-valued matrix D
        such that:
            * np.abs(D[f,t]) is the magnitude of frequency bin f at frame t
            * np.angle(D[f,t]) is the phase of frequency bin f at frame t
        This returns an amplitude-scaled Spectrogram (not dB-scaled)
    '''
    n_fft, hop_length, win_length = _stft_params()
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, 
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
    n_fft = hparams.n_fft
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length



# Conversions

_mel_basis = None

def _amp_to_db(x):
    '''
        Converts a amplitude-scaled spectrogram to a
        dB-scaled spectrogram
    '''
    return 20 * np.log10(np.maximum(1e-5, x))

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
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    '''
        Creates a filterbank matrix to combine FFT bins into
        mel-frequency bins
    '''
    return librosa.filters.mel(hparams.sample_rate, 
        hparams.n_fft, n_mels=hparams.num_mels)

def _normalize(S):
    '''
        Input
        S: Spectrogram

        Returns a normalized version of the spectrogram.
        Since we don't care about absolute volume and only
        care about relatve volume, we pin the spectrogram frequency

    '''
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)