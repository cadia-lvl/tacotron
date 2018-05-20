import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

from hparams import hparams
from model.modules import cbhg, prenet


class Encoder:
    def __init__(self, is_training=False):
        self._hparams = hparams
        self._is_training = is_training

    def encode(self, inputs, input_lengths):
        '''
            Encode the embedded inputs by stepping through:
                1) Prenet
                2) The CBHG module

            param:
                inputs: The embedded inputs, shape = [N (32), T_in, embed_depth (256)]
                where N is the batch size and T_in is the maximum length of this batch
                inputs
        '''
        return self._cbhg(self._prenet(inputs), input_lengths)
    
    def _prenet(self, inputs):
        '''
            The prenet is the first module the input data is passed through. The prenet serves as
            a bottleneck layer and helps convergence and improve generaliztaion.
            
            params:
                inputs: embedded batch tensor [batch_size, padded_sequence_length, embed_depth]

            return:
                inputs after propagating throught the prenet
        '''
        return prenet(inputs, self._is_training, self._hparams.get('prenet_depths'))

    def _cbhg(self, inputs, input_lengths):
        '''
            The CBHG is the next module that the input data is passed through, coming from
            the prenet. At this point, the inputs will be vectors of length n (128), the number of
            hidden nodes in the last fully connected layer in the prenet. 

            param:
                inputs: tensors of the (possibly modified) inputs
                input_lengths: original (pre-padding) length of the sentences
               
            return:
                inputs after cbhg module propagation
        '''
        kwargs = {
            'K': self._hparams.get('encoder_K'),
            'bank_num_filters': self._hparams.get('encoder_bank_num_filters'),
            'pooling_stride': self._hparams.get('encoder_pooling_stride'),
            'pooling_width': self._hparams.get('encoder_pooling_width'),
            'proj_num_filters': self._hparams.get('encoder_proj_num_filters'),
            'proj_filter_width': self._hparams.get('encoder_proj_filter_width'),
            'num_highway_layers': self._hparams.get('encoder_num_highway_layers'),
            'highway_depth': self._hparams.get('encoder_highway_depth'),
            'gru_num_cells': self._hparams.get('encoder_gru_num_cells')
        }
        return cbhg(inputs, input_lengths, self._is_training, 'encoder_cbhg', **kwargs)
