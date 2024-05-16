import tensorflow as tf
import neural_toolbox.utils as utils

from tensorflow.python.ops.init_ops import RandomUniform

class CBNAbtract(object):
    """
    Factory (Design pattern) to use cbn
    """

    def create_cbn_input(self, feature_maps):
        """
        This method is called every time conditional batchnorm is applied on cbn
        This factory enable to inject cbn to a pretrained resnet

        The good practice is to put the input of cbn (lstm embedding for instance) in the constructor.
        One may then use this variable in the create cbn.

        e.g.
        def __init__(self, lstm_state):
            self.lstm_state = lstm_state

        def create_cbn_input(feature_map):
            feat = int(feature_maps.get_shape()[3])
            delta_betas = tf.contrib.layers.fully_connected(lstm_state, num_outputs=feat)
            delta_gammas = tf.contrib.layers.fully_connected(lstm_state, num_outputs=feat)
            return delta_betas, delta_gammas

        :param feature_maps: (None,h,w,f)
        :return: deltas_betas, delta_gammas: (None, f), (None, f)
        """

        batch_size = int(feature_maps.get_shape()[0])
        heigh = int(feature_maps.get_shape()[1])
        width = int(feature_maps.get_shape()[2])
        feat = int(feature_maps.get_shape()[3])

        delta_betas = tf.zeros(shape=[batch_size, feat])  # Note that this does not compile (batch_size=None)
        delta_gammas = tf.zeros(shape=[batch_size, feat])

        return delta_betas, delta_gammas


class CBNfromLSTM(CBNAbtract):
    """
    Basic LSTM for CBN
    """
    def __init__(self, lstm_state, no_units, use_betas=True, use_gammas=True):
        self.lstm_state = lstm_state
        self.cbn_embedding_size = no_units
        self.use_betas = use_betas
        self.use_gammas = use_gammas


    def create_cbn_input(self, feature_maps):
        no_features = int(feature_maps.get_shape()[3])
        batch_size = tf.shape(feature_maps)[0]

        if self.use_betas:
            h_betas = utils.fully_connected(self.lstm_state,
                                            self.cbn_embedding_size,
                                            weight_initializer=RandomUniform(-1e-4, 1e-4),
                                            scope="hidden_betas",
                                            activation='relu')
            delta_betas = utils.fully_connected(h_betas, no_features, scope="delta_beta",
                                                weight_initializer=RandomUniform(-1e-4, 1e-4), use_bias=False)
        else:
            delta_betas = tf.tile(tf.constant(0.0, shape=[1, no_features]), tf.stack([batch_size, 1]))

        if self.use_gammas:
            h_gammas = utils.fully_connected(self.lstm_state,
                                             self.cbn_embedding_size,
                                             weight_initializer=RandomUniform(-1e-4, 1e-4),
                                             scope="hidden_gammas",
                                             activation='relu')
            delta_gammas = utils.fully_connected(h_gammas, no_features, scope="delta_gamma",
                                                 weight_initializer=RandomUniform(-1e-4, 1e-4))
        else:
            delta_gammas = tf.tile(tf.constant(0.0, shape=[1, no_features]), tf.stack([batch_size, 1]))

        return delta_betas, delta_gammas