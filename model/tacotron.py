import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from model import encoder, decoder, modules
from .hparams import hparams as hp
from model.rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper

class Tacotron:
    def __init__(self, hparams=hp):
        self.hparams = hparams

    def initialize(self, batch):
        '''
            param:
            inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
              steps in the input time series, and values are character IDs
            input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
              of each sequence in inputs.
            mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
              of steps in the output time series, M is num_mels, and values are entries in the mel
              spectrogram. Only needed for training.
            linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
              of steps in the output time series, F is num_freq, and values are entries in the linear
              spectrogram. Only needed for training.
        '''
        with tf.variable_scope('inference') as scope:
            is_training = linear_targets is not None
            # TODO: How is this not always the same number ? 
            batch_size = batch.get_size()

            # Encoder
            encoder = Encoder(is_training=is_training)
            encoder_outputs = encoder.encode(batch.get_embedds(), batch.get_input_lengths())

            # Attention
            attention_cell = AttentionWrapper(
                    DecoderPrenetWrapper(GRUCell(hp.attention_depth), is_training, self.hparams.prenet_depths),
                    BahdanauAttention(self.hparams.attention_depth, encoder_outputs),
                    alignment_history=True,
                    output_attention=False) 

            # Concat
            concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)

            # Decoder 
            decoder = Decoder(is_training=is_training)
            dec_out, final_dec_state = decoder.decode(concat_cell,batch_size)

            mel_outputs = tf.reshape(dec_out, [batch_size,-1, self.hparams.num_mels])

            # Post processing CHBG
            post_out = modules.post_cbhg(mel_outputs, hp.num_mels, training, self.hparams.postnet_depth)
            linear_out = tf.layers.dense(post_out, hp.num_freq)

            # Alignments
            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1,2,0])

            self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = batch.get_all()
            self.mel_outputs = batch.get_mel_outputs()
            self.linear_outputs = linear_out
            self.alignments = alignments

    def add_loss(self):
        '''
            Adds loss to the model. Sets "loss" field. initialize 
            must have been called.
        '''
        with tf.variable_scope('loss') as scope:
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            # Prioritize loss for frequencies under 3000 Hz.
            n_priority_freq = int(3000 / (self.hp.sample_rate * 0.5) * self.hp.num_freq)
            self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
            self.loss = self.mel_loss + self.linear_loss


    def add_optimizer(self, global_step):
        '''
            Adds optimizer. Sets "gradients" and "optimize" fields. 
            add_loss must have been called.
            
            Args:
                global_step: int32 scalar Tensor representing current global 
                step in training
        '''
        with tf.variable_scope('optimizer') as scope:
        if self.hparams.decay_learning_rate:
            self.learning_rate = _learning_rate_decay(self.hparams.initial_learning_rate, global_step)
        else:
            self.learning_rate = tf.convert_to_tensor(self.hparams.initial_learning_rate)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.adam_beta1, self.hparams.adam_beta2)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        self.gradients = gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
        # https://github.com/tensorflow/tensorflow/issues/1122
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
            global_step=global_step)

def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
