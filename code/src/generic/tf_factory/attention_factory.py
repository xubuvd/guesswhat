import tensorflow as tf

from neural_toolbox.attention import compute_attention, compute_glimpse

"""
get_attention(self.images, None, "none")
mytest:   self.images shape=[batch_size,VGG16_conv5_3=14,14,512]
"""
def get_attention(feature_map, lstm, config, dropout_keep=1, reuse=False):
    attention_mode = config.get("mode", None)

    if attention_mode == "none":
        image_out = feature_map

    elif attention_mode == "mean":
        image_out = tf.reduce_mean(feature_map, axis=(1, 2))

    elif attention_mode == "classic":
        image_out = compute_attention(feature_map,
                                      lstm,
                                      no_mlp_units=config['no_attention_mlp'],
                                      reuse=reuse)

    elif attention_mode == "glimpse":
        image_out = compute_glimpse(feature_map,
                                    lstm,
                                    no_glimpse=config['no_glimpses'],
                                    glimpse_embedding_size=config['no_attention_mlp'],
                                    keep_dropout=dropout_keep,
                                    reuse=reuse)
    else:
        assert False, "Wrong attention mode: {}".format(attention_mode)

    return image_out
