
import tensorflow as tf
import numpy as np

def append_spatial_location(features, min_max = list([-1,1])):

    with tf.variable_scope("spatial_location"):
        # Retrieve feature dimension
        batch_size = tf.shape(features)[0]
        height = int(features.get_shape()[1])
        width = int(features.get_shape()[2])

        # Create numpy spatial array
        h_array = np.tile(np.linspace(min_max[0], min_max[1], num=height), (width,1))
        w_array = np.tile(np.linspace(min_max[0], min_max[1], num=width), (height,1))
        spatial_array = np.stack( (np.transpose(h_array), w_array), axis=2)

        # Create spatial feature as a tensorflow constant and tile with batch size
        spatial_feat = tf.constant(spatial_array, dtype=tf.float32, shape=spatial_array.shape)
        spatial_feat = tf.expand_dims(spatial_feat, axis=0)
        spatial_feat = tf.tile(spatial_feat, tf.stack([batch_size, 1, 1, 1]))

    # Append spatial features to the feature map
    features = tf.concat([features, spatial_feat], axis=3)

    return features



def pooling_to_shape(feature_maps, shape, pooling=tf.nn.avg_pool):
    cur_h = int(feature_maps.get_shape()[1])
    cur_w = int(feature_maps.get_shape()[2])

    if cur_h > shape[0] and cur_w > shape[1]:
        stride_h, stride_w = int(cur_h / shape[0]), int(cur_w / shape[1])
        reduce_fm = pooling(feature_maps, ksize=[1, stride_h, stride_w, 1], strides=[1, stride_h, stride_w, 1], padding="VALID")
    else:
        reduce_fm = feature_maps

    return reduce_fm