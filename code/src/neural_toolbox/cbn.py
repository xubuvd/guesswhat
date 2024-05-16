import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers.python.layers import utils as tf_utils
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops


class ConditionalBatchNorm(object):
    """
    Batch normalization layer that may take a cbn factory as input
    """

    def __init__(self, cbn_input_builder, epsilon=1e-5, is_training=True, excluded_scope_names=list()):
        """
        :param cbn_input_builder: user defined cbn factory
        :param epsilon: epsilon for batch norm (stability)
        :param is_training: training/testing time
        :param excluded_scope_names: do not apply CBN if the current scope name is in the list
        """
        self.cbn_input_builder = cbn_input_builder
        self.epsilon = epsilon
        self.is_training = is_training
        self.excluded_scope_names = excluded_scope_names

    def apply(self, input):
        """
        When this fucntion overide the classic tf.batch_norm call
        We thus retrieve the initial gammas/betas and update them according the cbn_input_builder
        :param input:
        :return: input after been modulated by CBN
        """
        num_feature_maps = int(input.get_shape()[3])
        batch_size = tf.shape(input)[0]

        # Set variable scope similar to Slim resnet
        # such that the *same* variables are fetched when a pretrained Resnet is loaded
        with tf.variable_scope("BatchNorm"):
            betas = tf.get_variable("beta", [num_feature_maps], dtype=tf.float32)
            gammas = tf.get_variable("gamma", [num_feature_maps], dtype=tf.float32)

        betas = tf.tile(tf.expand_dims(betas, 0), tf.stack([batch_size, 1]))
        gammas = tf.tile(tf.expand_dims(gammas, 0), tf.stack([batch_size, 1]))

        scope_name = tf.get_variable_scope().name
        exclude_cbn = any([(needle in scope_name) for needle in self.excluded_scope_names])
        exclude_cbn = exclude_cbn or "*" in self.excluded_scope_names

        delta_betas = tf.tile(tf.constant(0.0, shape=[1, num_feature_maps]), tf.stack([batch_size, 1]))
        delta_gammas = tf.tile(tf.constant(0.0, shape=[1, num_feature_maps]), tf.stack([batch_size, 1]))

        if not exclude_cbn:
            with tf.variable_scope("cbn_input"):
                print("CBN : {} - {}".format(input.name, input.get_shape()))
                feature_maps = tf.stop_gradient(input)

                delta_betas, delta_gammas = self.cbn_input_builder.create_cbn_input(feature_maps)

        with tf.variable_scope("joint"):
            betas += delta_betas
            gammas += delta_gammas

        out = batch_norm(
            input,
            gammas=gammas,
            betas=betas,
            epsilon=self.epsilon,
            is_training=self.is_training)
        return out



def batch_norm(input, gammas, betas, epsilon, is_training):
    """

    BatchNorm implementation with sample-specific beta and gamma parameters
    i.e. the shift and scaling parameters are different across a batch of examples

    :param input: feature map input. 3-D vector (+ batch size)
    :param gammas: BN gamma parameters. 1-D vector (+ batch size)
    :param betas: BN betas parameters. 1-D vector (+ batch size)
    :param epsilon: BN epsilon for stability
    :param is_training: compute (True) or use (False) moving mean and variance
    :return: input after BN
    """

    assert (len(input.get_shape()) == 4)
    num_channels = int(input.get_shape()[3])

    # use cbn input score to not initialize the variable with resnet values
    with tf.variable_scope("cbn_input"):
        moving_mean = tf.get_variable("moving_mean", [num_channels], dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable("moving_variance", [num_channels], dtype=tf.float32, trainable=False)

    def _training():
        """
        Internal function that delay updates moving_vars if is_training.
        """
        mean, variance = tf.nn.moments(input, [0, 1, 2])

        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.99, zero_debias=True)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.99, zero_debias=False)

        return mean, variance, update_moving_mean, update_moving_variance

    def _inference():
        return moving_mean, moving_variance, moving_mean, moving_variance

    # Collect mean/variance to prepare moving mean/variance
    means, variances, update_mean, update_variance = tf_utils.smart_cond(is_training,
                                                                         _training,
                                                                         _inference)

    # Add moving mean/variance to tue update_ops (cf tensorflow batchnorm documentation)
    updates_collections = ops.GraphKeys.UPDATE_OPS
    ops.add_to_collections(updates_collections, update_mean)
    ops.add_to_collections(updates_collections, update_variance)

    # apply batch norm
    inv = gammas * tf.expand_dims(tf.rsqrt(variances + epsilon), 0)
    expanded_inv = tf.reshape(inv, [-1, 1, 1, num_channels])
    expanded_mean = tf.reshape(means, [-1, 1, 1, num_channels])
    expanded_betas = tf.reshape(betas, [-1, 1, 1, num_channels])
    out = expanded_inv * (input - expanded_mean) + expanded_betas

    return out


# Test
if __name__ == '__main__':

    inp = tf.placeholder(tf.float32, [50, 5, 5, 10])
    is_training = tf.placeholder(tf.bool, name='is_training')

    gammas = tf.constant(np.ones((50, 10), dtype='float32'))
    betas = tf.constant(np.zeros((50, 10), dtype='float32'))

    out = batch_norm(inp, gammas, betas, 1e-5, is_training)

    mean, variance = tf.global_variables()[:2]

    # Warning : require to update moving mean/variance
    up_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update = control_flow_ops.with_dependencies([tf.group(*up_ops)], out)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            o = sess.run(update, feed_dict={inp: np.random.normal(loc=5.0, size=[50, 5, 5, 10]), is_training: True})
            m = sess.run(mean)

        for _ in range(10):
            o = sess.run(update, feed_dict={inp: np.random.normal(loc=5.0, size=[50, 5, 5, 10]), is_training: False})
            m = sess.run(mean)
