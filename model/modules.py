import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

def prenet(inputs, training, layers, scope=None):
    '''
    This prenet consists of 4 layers in linear fashion and in this work the 
    layers are defined as:
        1) Fully-connected layer, ReLU activation, m (256) hidden nodes
        2) Dropout-layer, dropout rate: 0.5
        3) Fully connected layer, ReLU activation, n (128) hidden nodes
        4) Dropout-layer, dropout rate: 0.5
    where m is equal to the dimension of the character-embedding (256) and n is much smaller than m.
    [m, n] is defined as prenet_depth in hparams
    
    param:
        inputs: embedded batch tensor [batch_size, padded_sequence_length, embed_depth]
        training: tf is training (boolean)
        layers: list of layer sizes
        scope: the tf variable scope

    return:
        inputs after propagating throught the prenet
    '''
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layers):
            dense = tf.layers.dense(inputs, units=size, activation=tf.nn.relu, name='prenet_dense_{}'.format(i+1))
            inputs = tf.layers.dropout(dense, training=training, name='prenet_dropout_{}'.format(i+1))
    return inputs

def cbhg(inputs, input_lengths, training, scope, **kwargs):
    '''
        The CBHG module consists of the following steps:
            1) Bank with K sets of 1-D convolutional filters where the k-th set contains C_k (128)
                filters of width k, each capturing differently granular local and contextual
                information (think 1-gram, 2-gram ... K-grams)
            2) The output from each set in the convolution bank is stacked together and max-pooled
                along-time, stride = 1 to preserve original time resolution.
            3) Then the output is passed through two fixed-size 1d-convolutional layers [128, 256] to
                match up the dimensionality of the input
            4) Next, a residual connection adds together the output at this point and the original
                input
            5) The output is then passed through a muli-layer (4) highway network to extract high-level
                features.
            6) Finally, a bi-directional GRU RNN is stacked on top to extract sequential features from
                forward and backward context.

        TODO: Update documentation here about kwargs
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
        # 1-D convolution banking. Concatenate on the last axis (iterating filters) so that 
        # all filters from all sets are stacked
        with tf.variable_scope('conv_bank'):
            conv_bank_out = tf.concat(
                [convolute(inputs, k, kwargs.get('bank_num_filters'), tf.nn.relu, training, 
                    'conv_bank_{}'.format(k)) for k in range(1,kwargs.get('K')+1)],
                axis=-1)

        # Max-pool
        maxpool_out = tf.layers.max_pooling1d(conv_bank_out,pool_size=kwargs.get('pooling_width'),
            strides=kwargs.get('pooling_stride'),padding='same')

        # Projection layers
        proj1_out = convolute(maxpool_out, kwargs.get('proj_filter_width'), 
            kwargs.get('proj_num_filters')[0], tf.nn.relu, training, 'proj_1')
        proj2_out = convolute(proj1_out, kwargs.get('proj_filter_width'),
            kwargs.get('proj_num_filters')[1], None, training, 'proj_2')

        # combine the output with the original input in this residual connection
        highway_in = proj2_out + inputs

        # TODO: what the duck is this? 
        if highway_in.shape[2] != kwargs.get('highway_depth'):
            highway_in = tf.layers.dense(highway_in, kwargs.get('highway_depth'))

        # HighwayNet
        for i in range(kwargs.get('num_highway_layers')):
            highway_in = highwaynet(highway_in, 'highway_{}'.format(i+1), kwargs.get('highway_depth'))
        rnn_in = highway_in

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(kwargs.get('gru_num_cells')),
            GRUCell(kwargs.get('gru_num_cells')),
            rnn_in,
            sequence_length=input_lengths,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)

def highwaynet(inputs, scope, depth):
    '''
    A HighwayNet layer
    Given the inputs x, the output of the highway layer is described by
        y = H(x) * T(x) + x * C(x)
    where:
        H : Most often a ReLU layer (ReLU(Wx + b))
        T (transform gate): Here a sigmoid-activation layer
        C (carry gate): Here C(x) = 1 - T(x)


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
