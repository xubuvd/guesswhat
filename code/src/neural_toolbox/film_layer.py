import tensorflow as tf
import tensorflow.contrib.slim as slim

import neural_toolbox.ft_utils as ft_utils

def film_layer(ft, context, reuse=False):
    """
    A very basic FiLM layer with a linear transformation from context to FiLM parameters
    :param ft: features map to modulate. Must be a 3-D input vector (+batch size)
    :param context: conditioned FiLM parameters. Must be a 1-D input vector (+batch size)
    :param reuse: reuse variable, e.g, multi-gpu
    :return: modulated features
    """

    height = int(ft.get_shape()[1])
    width = int(ft.get_shape()[2])
    feature_size = int(ft.get_shape()[3])

    film_params = slim.fully_connected(context,
                                       num_outputs=2 * feature_size,
                                       activation_fn=None,
                                       reuse=reuse,
                                       scope="film_projection")

    film_params = tf.expand_dims(film_params, axis=[1])
    film_params = tf.expand_dims(film_params, axis=[1])
    film_params = tf.tile(film_params, [1, height, width, 1])

    gammas = film_params[:, :, :, :feature_size]
    betas = film_params[:, :, :, feature_size:]

    output = (1 + gammas) * ft + betas

    return output


class FiLMResblock(object):
    def __init__(self, features, context, is_training,
                 film_layer_fct=film_layer,
                 kernel1=list([1, 1]),
                 kernel2=list([3, 3]),
                 conv_weights_regularizer=None,
                 spatial_location=True, reuse=None):

        # Retrieve the size of the feature map
        feature_size = int(features.get_shape()[3])

        # Append a mask with spatial location to the feature map
        if spatial_location:
            features = ft_utils.append_spatial_location(features)

        # First convolution
        self.conv1_out = slim.conv2d(features,
                                 num_outputs=feature_size,
                                 kernel_size=kernel1,
                                 activation_fn=tf.nn.relu,
                                 weights_regularizer=conv_weights_regularizer,
                                 scope='conv1',
                                 reuse=reuse)

        # Second convolution
        self.conv2 = slim.conv2d(self.conv1_out,
                                 num_outputs=feature_size,
                                 kernel_size=kernel2,
                                 activation_fn=None,
                                 weights_regularizer=conv_weights_regularizer,
                                 scope='conv2',
                                 reuse=reuse)

        # Center/reduce output (Batch Normalization with no training parameters)
        self.conv2_bn = slim.batch_norm(self.conv2,
                                  center=False,
                                  scale=False,
                                  trainable=False,
                                  is_training=is_training,
                                  scope="bn",
                                  reuse=reuse)

        # Apply FILM layer Residual connection
        with tf.variable_scope("FiLM", reuse=reuse):
            self.conv2_film = film_layer_fct(self.conv2_bn, context, reuse=reuse)

        # Apply ReLU
        self.conv2_out = tf.nn.relu(self.conv2_film)

        # Residual connection
        self.output = self.conv2_out + self.conv1_out

    def get(self):
        return self.output


if __name__ == '__main__':
    import numpy as np

    feature_maps = tf.placeholder(tf.float32, shape=[None, 3, 3, 2])
    lstm_state = tf.placeholder(tf.float32, shape=[None, 6])

    modulated_feat1 = film_layer(ft=feature_maps, context=lstm_state)
    modulated_feat2 = FiLMResblock(features=feature_maps, context=lstm_state, is_training=True).get()

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_variables(tf.trainable_variables()))

    feature_maps_np = np.array(
    [
        [
          [ [1,2,3  ], [1, 2, 3] , [0, 0, 0]],
          [ [1, 2,3 ], [1, 2, 3] , [1, 1, 1] ]
        ],
        [
            [[-1, 1, 0], [0, 0, 0], [-1, 1, 0]],
            [[-1, 1, 0], [-1, 1, 1], [-4, 1, 4]]
        ]
    ]    )
    feature_maps_np = np.transpose(feature_maps_np, axes=[0,2,3,1])

    feature_maps_cst = tf.constant(feature_maps_np, dtype=tf.float32)
    lstm_state_cst = tf.constant( np.array([[1,0,1,0,1, 0], [0, 1, 1, 0, 1, 0 ]]), dtype=tf.float32)

    modulated_feat_cst = film_layer(feature_maps_cst, lstm_state_cst, reuse=True)
    print(modulated_feat_cst)
