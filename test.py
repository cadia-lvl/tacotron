import numpy as np
import matplotlib.pyplot as plt
import audio_tools as at
import io
from hparams import hparams
import librosa

spect = np.load('training/ljspeech-mel-00000.npy')
plt.pcolormesh(spect.T)
plt.show()


# This freaks out, issues with the GLA implementation
spect = np.load('training/ljspeech-spec-00000.npy')
wav = at.spectrogram_inv(spect)
wav *= 32767 / max(0.01, np.max(np.abs(wav)))
librosa.output.write_wav('test.wav', wav.astype(np.int16), hparams.sample_rate)