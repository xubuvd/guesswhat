import tensorflow as tf


# For some reason, it is faster than MultiCell on tf
def variable_length_LSTM(inp, num_hidden, seq_length,
                         dropout_keep_prob=1.0, scope="lstm", depth=1,
                         layer_norm=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        states = []
        last_states = []
        rnn_states = inp
        for d in range(depth):
            with tf.variable_scope('lstmcell'+str(d)):

                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    num_hidden,
                    layer_norm=layer_norm,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse)

                rnn_states, rnn_last_states = tf.nn.dynamic_rnn(
                    cell,
                    rnn_states,
                    dtype=tf.float32,
                    sequence_length=seq_length,
                )
                states.append(rnn_states)
                last_states.append(rnn_last_states.h)

        states = tf.concat(states, axis=2)
        last_states = tf.concat(last_states, axis=1)

        return last_states, states

