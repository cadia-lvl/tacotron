import argparse
import os
import tensorflow as tf
from data.data_feed import SimpleDataFeeder
from datetime import datetime
from .hparams import hparams

def test_feeder(in_dir):
    feeder = SimpleDataFeeder(in_dir)
    feeder.get_next_superbatch()

def embed_batch(batch,alph_length):
    '''
    param:
        batch: a list of tuples where the first element of the tuple is a list of one-hot-id's for a sentence
        alph_length: the number of letters in the current alphabet (should be changed in future)

    return: [batch_size, padded_sentence_length, embed_depth] tensor representing the embedded sentences in the batch
    '''
    one_hots = [l[0] for l in batches]
    batch_tensor = tf.convert_to_tensor(np.array(one_hots),dtype=int32)
    embedding_table = tf.get_variable('embed_table',shape=(alph_length,hparams.embed_depth), \
                                      dtype=float32,intializer=tf.truncated_normal_initializer(stddev=0.5))
    embedded_inputs = tf.nn.embedding_lookup(embedding_table,batch_tensor)

def embed_superbatch(superbatch,alph_length):
    '''
    param:
        superbatch: a list of batches
        alph_length: the number of letters in the current alphabet (should be changed in future)

    return: list of embedded batches
    '''
    return [embed_batch(batch,alph_length) for batch in superbatch]



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron_data'))
  parser.add_argument('--input', default='training_demo')
  parser.add_argument('--name', default='tacotron_simple')
  args = parser.parse_args()
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  test_feeder(os.path.join(args.base_dir, args.input))

if __name__ == '__main__':
  main()
