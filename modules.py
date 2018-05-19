import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell


def prenet(inputs, training, layers, scope=None):
    '''
    param:
        inputs: embedded batch tensor [batch_size, padded_sequence_length, embed_depth]
        training: tf is training
        layers: list of layer sizes
        scope: the tf variable scope

    return:
        inputs after propagating throught the prenet
    '''
    dropout_rate = 0.5 if training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layers):
            dense = tf.layers.dense(inputs, units=size, activation=tf.nn.relu, name='prenet_dense_{}'.format(i+1))
            inputs = tf.layers.dropout(dense, rate=dropout_rate, name='prenet_dropout_{}'.format(i+1))
    return inputs

def cbhg(inputs, input_lengths, training, scope, K, proj_channels, depth):
    '''
    param:
        inputs: tensors of the (possibly modified) inputs
        input_lengths: original (pre-padding) length of the sentences
        training: tf is training
        scope: the tf variable scope
        K: convolutional frame
        proj_channels: channels for the projection convolution
        depth: ??

    return:
        inputs after cbhg module propagation
    '''
    with tf.variable_scope(scope):
        # 1-D convolution banking
        with tf.variable_scope('conv_bank'):
            conv_bank_out = tf.concat(
                [convolute(inputs, k, 128, tf.nn.relu, training, 'conv_bank_{}'.format(k)) for k in range(1,K+1)],
                axis=-1)

        # Max-pool
        maxpool_out = tf.layers.max_pooling1d(conv_bank_out,pool_size=2,strides=1,padding='same')

        # Projection layers
        proj1_out = convolute(maxpool_out, 3, proj_channels[0], tf.nn.relu, training, 'proj_1')
        proj2_out = convolute(proj1_out, 3, proj_channels[1], None, training, 'proj_2')

        highway_in = proj2_out + inputs

        highway_depth = depth // 2

        # HighwayNet
        for i in range(4):
            highway_in = highwaynet(highway_in, 'highway_{}'.format(i+1), highway_depth)
        rnn_in = highway_in

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(highway_depth),
            GRUCell(highway_depth),
            rnn_in,
            sequence_length=input_lengths,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)

def highwaynet(inputs, scope, depth):
    '''
    A HighwayNet layer

    param:
        inputs: tensor of inputs to the net
        scope: tf variable scope
        depth: size of layer

    return:
        returns the output of the highwaynet
    '''
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs, units=depth, activation=tf.nn.relu, name='H')
        T = tf.layers.dense(inputs, units=depth, activation=tf.nn.sigmoid, name='T',
                            bias_initializer=tf.constant_initializer(-1.0))
        return H*T + inputs*(1.0-T)

def convolute(inputs, K, channels, activation, training, scope):
    '''
    param:
        inputs: input tensors
        K: kernel size
        channels: channel size
        activation: activation function
        training: tf is training
        scope: tf variable scope

    return: output of the convolutional layer
    '''
    with tf.variable_scope(scope):
        conv_out = tf.layers.conv1d(inputs, filters=channels, kernel_size=K,
                                    activation=activation, padding='same')
    return tf.layers.batch_normalization(conv_out,training=training)

def encoder_cbhg(inputs, input_lengths, training, depths):
    '''
    param:
        inputs: input tensor
        input_lengths: original (pre-padding) length of the sentences
        training: tf is training
        depth: ??

    return: return from cbhg
    '''
    input_channels = inputs.get_shape()[2]
    return cbhg(inputs, input_lengths, training, scope='encoder_cbhg', K=16, proj_channels=[128, input_channels], depth=depth)

def post_cbhg(inputs, input_dim, training, depth):
    '''
    param:
        inputs: input tensors
        input_dim: size of output
        training: tf is training
        depth: ??

    return: return from cbhg
    '''
    return cbhg(inputs,None,training,scope='post_cbhg', K=8, proj_channels=[256,input_dim],depth=depth)

