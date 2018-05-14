import librosa
import numpy as np
from scipy import signal
from hparams import hparams

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))
    
def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)

def preemphasis(x):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)

def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length

if __name__ == '__main__':
    spectrogram('atli er her hahahahaha')