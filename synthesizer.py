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
  def load(self, checkpoint_path, restore_step, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    b = Batch((inputs, input_lengths, None, None), prep=False)
    with tf.variable_scope('model') as scope:
        self.model = Tacotron(hparams=hparams)
        self.model.initialize(b)
        self.wav_output = audio.spectrogram_tensorflow_inv(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    # Restore from a checkpoint if the user requested it.
    restore_path = '%s-%d' % (checkpoint_path, restore_step)
    saver = tf.train.Saver()
    saver.restore(self.session, restore_path)


  def synthesize(self, text):
    seq = text_to_onehot(text, 'basic_cleaners')
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.pre_emphasis_inv(wav)
    wav = wav[:audio.find_endpoint(wav)]
    # TODO: A path can be set here to save the .wav to disk
    out = '/home/atli/example.wav'
    audio.save_wav(wav, out)
    # return out.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron_data'))
    parser.add_argument('--model', default='silenced')

    args = parser.parse_args()
    run_name = args.model


    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    with tf.variable_scope('model') as scope:
        s = Synthesizer()
        s.load(checkpoint_path, 186000)
        s.synthesize('Þetta er alveg ótrúlegt! Ég skil hvað þú segir loksins!')
