import os
import numpy as np
import argparse
import random
from text.text_tools import text_to_onehot
from hparams import hparams
import threading
import tensorflow as tf
from batch import Batch


class DataFeeder(threading.Thread):
    '''
        Feeds batches from the dataset that has been
        generated at the in_dir path
    '''

    def __init__(self, coordinator, in_dir):
        super(DataFeeder, self).__init__()
        self._coordinator = coordinator
        self._in_dir = in_dir
        self._metadata = load_metadata(os.path.join(in_dir, 'train.txt'))
        random.shuffle(self._metadata)
        self._cursor = 0 # index of the next sample
        self._num_samples = len(self._metadata)
        self._hparams = hparams
        self.batch_size = hparams.batch_size
        self.superbatch_size = hparams.superbatch_size
        self.outputs_per_step = hparams.outputs_per_step

        # Placeholders for inputs and targets.
        self._placeholders = [
            tf.placeholder(tf.int32, [None, None], 'inputs'),
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets')
        ]

        # Create queue of capacity 8 for buffering data which
        # will buffer 8 superbatches onto the FIFO queue
        queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32], name='input_queue')
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = queue.dequeue()
        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.mel_targets.set_shape(self._placeholders[2].shape)
        self.linear_targets.set_shape(self._placeholders[3].shape)


    def start_in_session(self, session):
        self._session = session
        self.start()


    def _enqueue_next_superbatch(self):
        '''
            Get the next superbatch (a list of batches). 
            The size of superbatches is set in hparams.
        '''
        superbatch =  [self._get_next_sample() for _ in range(self.superbatch_size*self.batch_size)]
        # sort the samples in the superbatch on length w.r.t. time
        superbatch.sort(key=lambda x: x[-1])
        # now bucket the batches in that order to improve efficiency
        batches = [superbatch[i:i+self.batch_size] for i in range(0, len(superbatch), self.batch_size)]
        random.shuffle(batches)
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, batch))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)


    def _get_next_sample(self):
        '''
            Loads a single sample from the dataset
            
            Output:
            (Onehot text input, mel target, linear target, cost)
        '''
        lin_target_path, mel_target_path, n_frames, text = self._metadata[self._cursor] 
        self.increment_cursor()
        lin_target = np.load(os.path.join(self._in_dir, lin_target_path))
        mel_target = np.load(os.path.join(self._in_dir, mel_target_path))
        onehot_text = text_to_onehot(text)
        return (onehot_text, mel_target, lin_target, n_frames)
    
    def increment_cursor(self):
        '''
            Increments the dataset cursor, or sets it
            to 0 if we have reached the end of the dataset
        '''
        if self._cursor >= self._num_samples:
            # start from beginning and shuffle the
            # data again
            self._cursor = 0
            random.shuffle(self._metadata)
        else:   
            self._cursor += 1


class SimpleDataFeeder:
    def __init__(self, in_dir):
        self._in_dir = in_dir
        self._metadata = load_metadata(os.path.join(in_dir, 'train.txt'))
        random.shuffle(self._metadata)
        self._cursor = 0 # index of the next sample
        self._num_samples = len(self._metadata)
        self._hparams = hparams
        self.batch_size = hparams.batch_size
        self.superbatch_size = hparams.superbatch_size


    def get_next_superbatch(self):
        '''
            Get the next superbatch (a list of batches). 
            The size of superbatches is set in hparams.
        '''
        superbatch =  [self._get_next_sample() for _ in range(self.superbatch_size*self.batch_size)]
        # sort the samples in the superbatch on length w.r.t. time
        superbatch.sort(key=lambda x: x[-1])
        # now bucket the batches in that order to improve efficiency
        batches = [Batch(superbatch[i:i+self.batch_size])
            for i in range(0, len(superbatch), self.batch_size)]
        random.shuffle(batches)
        return batches

    def _get_next_sample(self):
        '''
            Loads a single sample from the dataset
            
            Output:
            (Onehot text input, mel target, linear target, cost)
        '''
        lin_target_path, mel_target_path, n_frames, text = self._metadata[self._cursor] 
        self.increment_cursor()

        lin_target = np.load(os.path.join(self._in_dir, lin_target_path))
        mel_target = np.load(os.path.join(self._in_dir, mel_target_path))
        onehot_text = text_to_onehot(text)
        return (onehot_text, mel_target, lin_target, n_frames)

    def increment_cursor(self):
        '''
            Increments the dataset cursor, or sets it
            to 0 if we have reached the end of the dataset
        '''
        if self._cursor >= self._num_samples - 1:
            # start from beginning and shuffle the
            # data again
            self._cursor = 0
            random.shuffle(self._metadata)
        else:   
            self._cursor += 1

def load_metadata(path):
    '''
        Loads the metadata generated by the prep functions
        at the given path
    '''
    with open(path, encoding='utf-8') as f:
      metadata = [line.strip().split('|') for line in f]
      #hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
      #log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))
    return metadata