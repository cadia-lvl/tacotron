import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauAttention, BasicDecoder, dynamic_decode, Helper
from model.rnn_wrappers import ConcatOutputAndAttentionWrapper, DecoderPrenetWrapper
from hparams import hparams

class Decoder:
    # TODO
    def __init__(self, helper, is_training=False):
        self._hparams = hparams
        self._is_training = is_training
        self._helper = helper
    
    def decode(self, encoder_outputs, batch_size):
        # Attention
        attention_cell = AttentionWrapper(
            DecoderPrenetWrapper(GRUCell(self._hparams.attention_depth), self._is_training, self._hparams.prenet_depths),
            BahdanauAttention(self._hparams .attention_depth, encoder_outputs),
            alignment_history=True,
            output_attention=False)                                                  # [N, T_in, attention_depth=256]

        # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
        concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              # [N, T_in, 2*attention_depth=512]

        # Decoder (layers specified bottom to top):
        decoder_cell = MultiRNNCell([
            OutputProjectionWrapper(concat_cell, self._hparams.decoder_depth),
            ResidualWrapper(GRUCell(self._hparams.decoder_depth)),
            ResidualWrapper(GRUCell(self._hparams.decoder_depth))
            ], state_is_tuple=True)                                                  # [N, T_in, decoder_depth=256]

        # Project onto r mel spectrograms (predict r outputs at each RNN step):
        output_cell = OutputProjectionWrapper(decoder_cell, self._hparams.num_mels * self._hparams.outputs_per_step)
        decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        (decoder_outputs, _), final_decoder_state, _ = dynamic_decode(
            BasicDecoder(output_cell, self._helper, decoder_init_state),
            maximum_iterations=self._hparams.max_iters)                                         # [N, T_out/r, M*r]