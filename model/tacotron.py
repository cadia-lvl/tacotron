import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from model import encoder, decoder, modules
from .hparams import hparams as hp
from model.rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper
from .data.data_feeder import Datafeeder

class Tacotron:
    def __init__(self, hparams=hp):
        self._hparams = hparams

    def initialize(self, batch):
        '''
        param:
            batch: Batch object
        '''
        with tf.variable_scope('inference') as scope:
            inputs = batch._inputs
            input_lengths = batch._input_lengths
            mel_targets = batch._mel_targets
            linear_targets = batch._lin_targets
            training = linear_targes is not None
            batch_size = tf.shape(inputs)[0]

            # Encoder
            encoder = Encoder(is_training=is_training)
            encoder_outputs = encoder.encode(batch.get_embedds(), batch.get_input_lengths())

            # Attention
            attention_cell = AttentionWrapper(
                    DecoderPrenetWrapper(GRUCell(hp.attention_depth), is_training, self._hparams.prenet_depths),
                    BahdanauAttention(self._hparams.attention_depth, encoder_outputs),
                    alignment_history=True,
                    output_attention=False) 

            # Concat
            concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)

            # Decoder 
            decoder = Decoder(is_training=is_training)
            dec_out, final_dec_state = decoder.decode(concat_cell,batch_size)

            mel_outputs = tf.reshape(dec_out, [batch_size,-1, self._hparams.num_mels])

            # Post processing CHBG
            post_out = modules.post_cbhg(mel_outputs, hp.num_mels, training, self._hparams.postnet_depth)
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
        if self._hparams.decay_learning_rate:
            self.learning_rate = _learning_rate_decay(self._hparams.initial_learning_rate, global_step)
        else:
            self.learning_rate = tf.convert_to_tensor(self._hparams.initial_learning_rate)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.adam_beta1, self._hparams.adam_beta2)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        self.gradients = gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
        # https://github.com/tensorflow/tensorflow/issues/1122
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
            global_step=global_step)

    def train(self, log_dir, args):
        checkpoint_path = os.path.join(log_dir, 'model.ckpt')
        input_path = os.path.join(args.base_dir, args.input)

        # Coordinator and Datafeeder
        coord = tf.train.Coordinator()
        with tf.variable_scope('datafeeder') as scope:
            feeder = Datafeeder(coord,input_path)

        



def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
