import os
import argparse
import time
import traceback
import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell, MultiRNNCell,OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauAttention, BasicDecoder
from model.helpers import TestingHelper, TrainingHelper
from model import modules
from model.encoder import Encoder
from model.decoder import Decoder
import tools.audio
from text.text_tools import onehot_to_text

from hparams import hparams


from data.data_feed import DataFeeder
from tools import audio, logger, ValueWindow

class Tacotron:
    def __init__(self, hparams=hparams):
        self._hparams = hparams

    def initialize(self, batch):
        '''
        param:
            batch: Batch object
        '''
        with tf.variable_scope('inference') as scope:
            linear_targets = batch._lin_targets
            is_training = linear_targets is not None
            batch_size = batch.get_size()
            
            # Encoder
            encoder = Encoder(is_training=is_training)
            encoder_outputs = encoder.encode(batch.get_embedds(), batch.get_input_lengths())

            # Decoder
            if is_training:
                helper = TrainingHelper(batch.get_inputs(), batch.get_mel_targets(), 
                    self._hparams.num_mels, self._hparams.outputs_per_step)
            else:
                helper = TestingHelper(batch_size, self._hparams.num_mels, 
                    self._hparams.outputs_per_step)
            decoder = Decoder(helper, is_training=is_training)
            mel_outputs, lin_outputs, final_decoder_state = decoder.decode(encoder_outputs,batch_size)

           
            # Alignments
            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1,2,0])

            self.inputs, self.input_lengths, self.mel_targets, self.linear_targets = batch.get_all()
            self.mel_outputs = mel_outputs
            self.linear_outputs = lin_outputs
            self.alignments = alignments
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def add_stats(self):
        with tf.variable_scope('stats') as scope:
            tf.summary.histogram('linear_outputs', self.linear_outputs)
            tf.summary.histogram('linear_targets', self.linear_targets)
            tf.summary.histogram('mel_outputs', self.mel_outputs)
            tf.summary.histogram('mel_targets', self.mel_targets)
            tf.summary.scalar('loss_mel', self.mel_loss)
            tf.summary.scalar('loss_linear', self.linear_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            gradient_norms = [tf.norm(grad) for grad in self.gradients]
            tf.summary.histogram('gradient_norm', gradient_norms)
            tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
            return tf.summary.merge_all()
    
    
    def add_loss(self):
        '''
            Adds loss to the model. Sets "loss" field. initialize 
            must have been called.
        '''
        with tf.variable_scope('loss') as scope:
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            # Prioritize loss for frequencies under 3000 Hz.
            n_priority_freq = int(3000 / (self._hparams.sample_rate * 0.5) * self._hparams.num_freq)
            self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
            self.loss = self.mel_loss + self.linear_loss


    def add_optimizer(self):
        '''
            Adds optimizer. Sets "gradients" and "optimize" fields. 
            add_loss must have been called.
            
            Args:
                global_step: int32 scalar Tensor representing current global 
                step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            if self._hparams.decay_learning_rate:
                self.learning_rate = _learning_rate_decay(self._hparams.initial_learning_rate, self.global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(self._hparams.initial_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, self._hparams.adam_beta1, self._hparams.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                global_step=self.global_step)

    def train(self, log_dir, args):
        checkpoint_path = os.path.join(log_dir, 'model.ckpt')
        input_path = os.path.join(args.base_dir, args.input)
        #TODO: Fix this log path issue
        self._logger = logger.TrainingLogger(log_dir, slack_url=args.slack_url)

        # Coordinator and Datafeeder
        coord = tf.train.Coordinator()
        with tf.variable_scope('datafeeder') as scope:
            feeder = DataFeeder(coord,input_path, self._logger)

        with tf.variable_scope('model') as scope:
            self.initialize(feeder.current_batch)
            self.add_loss()
            self.add_optimizer()
            stats = self.add_stats()

        step = 0
        time_window = ValueWindow(100)
        loss_window = ValueWindow(100)
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

        # Train!
        with tf.Session() as sess:
            try:
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                sess.run(tf.global_variables_initializer())

                if args.restore_step:
                    # Restore from a checkpoint if the user requested it.
                    restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                    saver.restore(sess, restore_path)
                    self._logger.log('Resuming from checkpoint: %s' % restore_path, slack=True)
                else:
                    self._logger.log('Starting new training run', slack=True)

                feeder.start_in_session(sess)

                while not coord.should_stop():
                    start_time = time.time()
                    step, loss, opt = sess.run([self.global_step, self.loss, self.optimize])
                    time_window.append(time.time() - start_time)
                    loss_window.append(loss)
                    message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
                    step, time_window.average, loss, loss_window.average)
                    self._logger.log(message, slack=(step % args.checkpoint_interval == 0))

                    if step % args.summary_interval == 0:
                        self._logger.log('Writing summary at step: %d' % step, slack=False)
                        summary_writer.add_summary(sess.run(stats), step)

                    if step % args.checkpoint_interval == 0:
                        self._logger.log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                        saver.save(sess, checkpoint_path, global_step=step)
                        self._logger.log('Saving audio and alignment...')
                        input_seq, spectrogram, alignment = sess.run([
                            self.inputs[0], self.linear_outputs[0], self.alignments[0]])
                        waveform = audio.spectrogram_inv(spectrogram.T)
                        audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
                        #plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                        #    info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
                        self._logger.log('Input: %s' % onehot_to_text(input_seq))

            except Exception as e:
                #log.log('Exiting due to exception: %s' % e, slack=True)
                traceback.print_exc()
                coord.request_stop(e)


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

