import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from hparams import hparams
from modules import prenet, cbhg


class Decoder:
    # TODO