import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from hparams import hparams


class Decoder:
    # TODO
    def __init__(self, is_training=False):
        self._hparams = hparams
        self._is_training = is_training

    def decode(self, inputs,batch_size):
        self.decoder_cell = MultiRNNCell([
            OutputProjectionWrapper(inputs, self._hparams.decoder_depth),
            ResidualWrapper(GRUCell(self._hparams.decoder_depth)),
            ResidualWrapper(GRUCell(self._hparams.decoder_depth))
        ], state_is_tuple=True)

        self.output_cell = OutputProjectionWrapper(self.decoder_cell, hp.num_mels*hp.output_size)
        self.decoder_init_state = self.output_cell.zero_stat(batch_size=batch_size, dtype=tf.float32)
        if self._is_training:
            self.helper = ..
        else:
            self.helper = ..

        (self.decoder_outputs, _), self.final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(self.output_cell, helper, self.decoder_init_state),
        maximum_iterations=hp.max_iters)

        return self.decoder_outputs, self.final_decoder_state
