import numpy as np
import matplotlib.pyplot as plt
import audio_tools as at
import io
from hparams import hparams
import librosa

'''
spect = np.load('training/ljspeech-spec-00001.npy')
plt.pcolormesh(spect.T)
plt.show()
'''

# This freaks out, issues with the GLA implementation
spect = np.load('training/ljspeech-spec-01234.npy')
print(spect.shape)
wav = at.spectrogram_inv(spect.T)
wav *= 32767.0 / max(0.01, np.max(np.abs(wav)))
librosa.output.write_wav('test_1.wav', wav.astype(np.float16), 22050)
librosa.output.write_wav('test_2.wav', wav.astype(np.float16), 20000)
librosa.output.write_wav('test_3.wav', wav.astype(np.float16), 24000)