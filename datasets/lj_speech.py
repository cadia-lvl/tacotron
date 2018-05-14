from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import audio_tools as at


'''
    A python handler for the LJSpeech data-set

    About data-set:
    Metadata is provided in metadata.csv. This file consists of one record per
    line, delimited by the pipe character (0x7c). The fields are:

    1. ID: this is the name of the corresponding .wav file
    2. Transcription: words spoken by the reader (UTF-8)
    3. Normalized Transcription: transcription with numbers, ordinals, and
        monetary units expanded into full words (UTF-8).

    Each audio file is a single-channel 16-bit PCM WAV with a sample rate of
    22050 Hz.

    STATISTICS

    Total Clips            13,100
    Total Words            225,715
    Total Characters       1,308,674
    Total Duration         23:55:17
    Mean Clip Duration     6.57 sec
    Min Clip Duration      1.11 sec
    Max Clip Duration      10.10 sec
    Mean Words per Clip    17.23
    Distinct Words         13,821
'''

_sample_rate = 22050
_data_dir = os.path.join(os.getcwd(), 'datasets/LJSpeech-1.1/')

def load_data(out_dir):
    '''

    '''
    executor = ProcessPoolExecutor(max_workers = cpu_count())
    futures = []
    index = 0
    with open(os.path.join(_data_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
           # Each line has the form "{id} | {text}"
           # where {id} indexes the relevant .wav file
           [wav_id, text, norm_text] = line.strip().split('|')
           futures.append(executor.submit(partial(_process_utterance, out_dir, wav_id, index, norm_text)))
           index += 1
    return [future.result() for future in tqdm(futures)]  

def load_wav(path):
  return librosa.core.load(path, sr=_sample_rate)[0]

def _process_utterance(out_dir, wav_id, index, text):
    wav_path = os.path.join(_data_dir, 'wavs', '%s.wav' % wav_id)
    wav = load_wav(wav_path)
    # Get the linear-scale spectrogram
    spect = at.spectrogram(wav).astype(np.float32)
    n_frames = spect.shape[1]
    # Get the mel-scaled spectrogram
    melspect = at.mel_spectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spect_filename = 'ljspeech-spec-%05d.npy' % index
    melspect_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spect_filename), spect.T, allow_pickle=False)
    np.save(os.path.join(out_dir, melspect_filename), melspect.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spect_filename, melspect_filename, n_frames, text)
