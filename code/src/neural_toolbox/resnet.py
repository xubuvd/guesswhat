import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
import os

def get_resnet_arg_scope(bn_fn):
    """
    Trick to apply CBN from a pretrained tf network. It overides the batchnorm constructor with cbn
    :param bn_fn: cbn factory
    :return: tensorflow scope
    """

    with arg_scope(
            [layers_lib.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=bn_fn,
            normalizer_params=None) as arg_sc:
        return arg_sc


def create_resnet(image_input, is_training, scope="", resnet_out="block4", resnet_version=50, cbn=None):
    """
    Create a resnet by overidding the classic batchnorm with conditional batchnorm
    :param image_input: placeholder with image
    :param is_training: are you using the resnet at training_time or test_time
    :param scope: tensorflow scope
    :param resnet_version: 50/101/152
    :param cbn: the cbn factory
    :return: the resnet output
    """

    if cbn is None:
        # assert False, "\n" \
        #               "There is a bug with classic batchnorm with slim networks (https://github.com/tensorflow/tensorflow/issues/4887). \n" \
        #               "Please use the following config -> 'cbn': {'use_cbn':true, 'excluded_scope_names': ['*']}"
        arg_sc = slim_utils.resnet_arg_scope(is_training=is_training)
    else:
        arg_sc = get_resnet_arg_scope(cbn.apply)

    # Pick the correct version of the resnet
    if resnet_version == 50:
        current_resnet = resnet_v1.resnet_v1_50
    elif resnet_version == 101:
        current_resnet = resnet_v1.resnet_v1_101
    elif resnet_version == 152:
        current_resnet = resnet_v1.resnet_v1_152
    else:
        raise ValueError("Unsupported resnet version")

    resnet_scope = os.path.join('resnet_v1_{}/'.format(resnet_version), resnet_out)

    with slim.arg_scope(arg_sc):
        net, end_points = current_resnet(image_input, 1000)  # 1000 is the number of softmax class

    if len(scope) > 0 and not scope.endswith("/"):
        scope += "/"

    print("Use: {}".format(resnet_scope))
    out = end_points[scope + resnet_scope]

    return out
