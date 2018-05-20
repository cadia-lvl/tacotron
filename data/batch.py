import numpy as np
from hparams import hparams
import random

class Batch:
    def __init__(self, data):
        random.shuffle(data)
        self._inputs = [x[0] for x in data]
        self._input_lengths = np.asarray([len(x[0]) for x in data], dtype=np.int32)
        self._mel_targets = [x[1] for x in data]
        self._lin_targets = [x[2] for x in data]
        self._prepare_batch()
            
    def _prepare_batch(self, outputs_per_step=hparams.outputs_per_step):
        '''
            Prepares both inputs and targets for
            inference
        '''
        self._prepare_inputs()
        self._prepare_targets()

    def _prepare_inputs(self):
        '''
            Find the maximum length of all input sequences, 
            pad every shorter sequence to that size and return 
            the padded inputs as a [N x max_len] matrix where 
            N = batch size resulting in e.g.
            
            --------------------------------
            | l_11, l_12,   0     0     0  | 1
            | l_21, l_22, l_23, l_24, l_25 | 2 
            | l_31, l_32, l_33, l_34    0  | 3
            -------------------------------- (x) 
               1     2     3     4     5 (char)

        '''
        max_len = max((len(x) for x in self._inputs))
        self._inputs = np.stack([pad_input(x, max_len) for x in self._inputs])
    
    def _prepare_targets(self):
        '''
            For both mel and linear targets:
            Find the spectrogram longest on the time axis, pad every target
            to the rounded length (up to outputs_per_step (5)). If e.g. the longest
            spectrogram is 8 long on the timeaxis we get a 3-step spectrogram to be

            --------------------------       
            | l_11, l_12, l_13  l_14 |  1
            | l_21, l_22, l_23, l_24 |  2
            | l_31, l_32, l_33, l_34 |  3
            |  0     0     0     0   |  4
            |  0     0     0     0   |  5
            |  0     0     0     0   |  6
            |  0     0     0     0   |  7
            |  0     0     0     0   |  8
            |  0     0     0     0   |  9
            |  0     0     0     0   |  10
            -------------------------- (t)
               1     2     3     4 (Hz)
            and all those spectrograms are stacked on top of one another on the first
            axis.

        '''
        max_len_lin = round_up(max([len(t) for t in self._lin_targets]) + 1)
        self._lin_targets =  np.stack([pad_target(t, max_len_lin) 
            for t in self._lin_targets])
        
        max_len_mel = round_up(max([len(t) for t in self._mel_targets]) + 1)
        self._mel_targets =  np.stack([pad_target(t, max_len_mel) 
            for t in self._mel_targets])

    def get_input_lengths(self):
        return self._input_lengths

def pad_input(x, length):
    '''
        Given a list, x, and an int, length, add length - len(x) 
        pad values to the back of the list and return a numpy vector
        
        Param:
            x: a list
            length: an integer
        
        Output:
            A padded numpy vector
    '''
    return np.pad(x, (0, length - x.shape[0]), mode='constant', 
        constant_values=hparams.pad_value)

def pad_target(t, length):
    '''
        Given an 2d array representing the target spectrogram where
        the first axis represents time and the second one frequency, and
        an integer, length, add length - len(time axis) pad values to
        the back of the time axis and return the array

        Param:
            t: a numpy 2d array
            length: an integer

        Output:
            A padded 2d numpy array

    '''
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', 
        constant_values=hparams.pad_value)

def round_up(x):
    '''
        Given an integer, x, round up x to the closest product of the
        outputs_per_step hyperparameter (5)

        Param:
            x: an integer

        Output:
            x rounded up to outputs_per_step

    '''
    remainder = x % hparams.outputs_per_step
    if remainder == 0:
          return x
    else:
        return x + hparams.outputs_per_step - remainder