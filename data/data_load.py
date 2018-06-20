import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import librosa
import numpy as np
from tqdm import tqdm

from hparams import hparams
from tools import audio


def prep_ljspeech(in_dir, out_dir):
    '''
        Preprocesses the LJ Speech dataset from a given input path into a 
        given output directory.

        Input:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into

        Output:
        A list of tuples describing the training examples.
    '''
    executor = ProcessPoolExecutor(max_workers = cpu_count())
    futures = []
    index = 0
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            # Each line has the form "{line_id} | {text} | {norm text}"
            # where {line_id} indexes the relevant .wav file and norm text
            # is a normalized version of the text (3 -> three etc.)
            [line_id, text, norm_text] = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % line_id)
            futures.append(executor.submit(partial(_process_utterance, out_dir, 
                index, wav_path, text, prefix='ljspeech')))
            index += 1
    return [future.result() for future in tqdm(futures)]

def prep_icelandic(in_dir, out_dir):
  '''
    Preprocesses the Icelandic dataset from a given input path into a given 
    output directory.

    Input:
      out_dir: The directory to write the output into
    Returns:
      A list of tuples describing the training examples.
  '''
  executor = ProcessPoolExecutor(max_workers=cpu_count())
  futures = []
  index = 1
  with open(os.path.join(in_dir, 'line_index.tsv'), encoding='utf-8') as f:
    for line in f:
        # Each line has the form "{line_id} \t {owner} \t {internal_id} \t {text}"
        # where {line_id} indexes the relevant .wav file and a text token which
        # is the same version of the text as {text}  
        [line_id, owner, intrnl_id, text] = line.strip().split('\t')
        wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % line_id)
        futures.append(executor.submit(partial(_process_utterance, out_dir, 
            index, wav_path, text, prefix='icelandic')))
        index += 1
  return [future.result() for future in tqdm(futures)]

def prep_ivona(in_dir, out_dir, trim_silence=False, outlier_index_path=None, index_path=None):
	'''
		Preprocesses the Ivona dataset from a given input path into a given 
		output directory.

		Input:
			* out_dir: The directory to write the output into
			* trim_silence: If true, silence is trimmed from both ends
			* outlier_index_path: If not none, outliers will not be preprocessed
				and therefore not used for training
		Returns:
			* A list of tuples describing the training examples.
	'''
	executor = ProcessPoolExecutor(max_workers=4)
	index = 1
	futures = []
	outlier_indx = []
	if outlier_index_path is not None:
		outlier_indx = load_outlier_indx(outlier_index_path)
	if index_path is None:
		index_path = os.path.join(in_dir, 'line_index.tsv')
	with open(index_path, encoding='utf-8') as f:
		for line in f:
			# Each line has the form "{token fname} \t {audio fname} \t {reader}"
			[token_fname, audio_fname] = line.strip().split('\t')
			if len(outlier_indx) > 0 and token_fname in outlier_indx:
				print(token_fname)
				outlier_indx.remove(token_fname)
			else:
				text = load_text(os.path.join(in_dir, 'ivona_txt', token_fname))
				wav_path = os.path.join(in_dir, 'Kristjan_export', '%s' % audio_fname)
				futures.append(executor.submit(partial(_process_utterance, out_dir, 
					index, wav_path, text, prefix='ivona', trim_silence=trim_silence)))
				index += 1
	return [future.result() for future in tqdm(futures)]

def load_text(path):
	text = ''
	with open(path, 'r') as f:
		for line in f:
			text += line
	return text

def load_outlier_indx(path):
	indx = []
	with open(path, 'r') as f:
		for line in f:
			indx.append(line.strip())
	return indx

def _process_utterance(out_dir, index, wav_path, text, prefix='data', trim_silence=False):
	'''
		Generates a linear and a mel spectrogram for the waveform
		file at the given wav_path.

		Input:
			* out_dir: The target directory for the generated spectrograms
			* index: Used for enumerating the generated files
			* wav_path: The path to the waveform file
			* text: The text being spoken

		Output:
			* The filename of the linear spectrogram
			* The filename of the mel spectrogram
			* The number of time-frames in the linear spectrogram
			* The text being spoken
	'''
	wav = audio.load_wav(wav_path)
	# Get the linear-scale spectrogram
	if trim_silence:
		wav = audio.trim_silence(wav)
	spect = audio.spectrogram(wav).astype(np.float32)
	n_frames = spect.shape[1]
	# Get the mel-scaled spectrogram
	melspect = audio.mel_spectrogram(wav).astype(np.float32)
	# Write the spectrograms to disk
	spect_filename = prefix+'-spec-%05d.npy' % index
	melspect_filename = prefix+'-mel-%05d.npy' % index
	np.save(os.path.join(out_dir, spect_filename), spect.T, allow_pickle=False)
	np.save(os.path.join(out_dir, melspect_filename), melspect.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (spect_filename, melspect_filename, n_frames, text)
