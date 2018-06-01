import argparse
import io
import os

import numpy as np
import tensorflow as tf
from librosa import effects

from data.batch import Batch
from hparams import hparams
from model.tacotron import Tacotron
from text.text_tools import text_to_onehot
from tools import audio


class Synthesizer:
  def load(self, checkpoint_dir, restore_step, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    # create a batch with a single input and no spectrograms
    b = Batch((inputs, input_lengths, None, None), prep=False)
    with tf.variable_scope('model') as scope:
        self.model = Tacotron(hparams=hparams)
        self.model.initialize(b)
        self.wav_output = audio.spectrogram_tensorflow_inv(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_dir)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    # Restore from a checkpoint if the user requested it.
    restore_dir = '%s-%d' % (checkpoint_dir, restore_step)
    saver = tf.train.Saver()
    saver.restore(self.session, restore_dir)


  def synthesize(self, text, synth_dir):
    seq = text_to_onehot(text, 'basic_cleaners')
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.pre_emphasis_inv(wav)
    wav = wav[:audio.find_endpoint(wav)]

    # create subfolders if not existing
    os.makedirs(os.path.join(synth_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(synth_dir, 'text'), exist_ok=True)
    index_file = open(os.path.join(synth_dir, 'index.txt'), 'a+')
    index = sum(1 for line in open(os.path.join(synth_dir, 'index.txt'))) + 1
    wav_path = os.path.join(synth_dir, 'wavs', 'synth-%03d.wav' % index)
    txt_path = os.path.join(synth_dir, 'text', 'text-%03d.txt' % index)
    index_file.write(txt_path+'|'+wav_path+'|'+text+'\n')
    txt_file = open(txt_path, 'w')
    txt_file.write(text)
    audio.save_wav(wav, wav_path)
    print('Sentence has been synthesized and is available at: ', wav_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', help='Select a valid checkpoint to synthesize from', default=0)
    parser.add_argument('--input_dir', help='relative path from /home/<user> to base training output directory'),
    parser.add_argument('--model_name', help='Select the name of the model')
    parser.add_argument('--text', help='Text to be synthesized')
    args = parser.parse_args()

    in_dir = os.path.expanduser('~/'+os.path.join(args.input_dir, args.model_name))
    meta_dir = os.path.join(in_dir, 'meta') # Location of model meta and checkpoints
    log_dir = os.path.join(in_dir, 'logs')
    synth_dir = os.path.join(in_dir, 'synthesized')
    os.makedirs(synth_dir, exist_ok=True) # create if synthesizing for the first time
    checkpoint_dir = os.path.join(meta_dir, 'model.ckpt')
    s = Synthesizer()
    s.load(checkpoint_dir, int(args.restore_step), model_name=args.model_name)
    s.synthesize(args.text, synth_dir)
