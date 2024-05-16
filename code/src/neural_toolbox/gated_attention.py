import tensorflow as tf


# https://arxiv.org/pdf/1606.01549.pdf

def gated_attention(feature, context):

    no_tokens = tf.shape(feature)[1]
    context = tf.expand_dims(context, axis=1)
    context = tf.tile(context, multiples=[1, no_tokens, 1, 1])

    context_t = tf.transpose(context,perm=(0,1,3,2))

    # Compute the softmax along the
    evidence = tf.matmul(context_t, tf.expand_dims(feature, axis=3))
    alpha  = tf.nn.softmax(evidence, dim=2)

    # Reduce context
    context_with_att = tf.matmul(context, alpha)
    context_with_att = tf.squeeze(context_with_att, axis=3)

    # Final gating
    gated_features = feature * context_with_att

    return gated_features


def gated_film(features, context, use_beta=True, use_softmax=False, keep_dropout=1, reuse=False):

    no_elem = int(features.get_shape()[1])
    feature_size = int(features.get_shape()[2])

    no_params = 1 if use_beta else 2

    context = tf.nn.dropout(context, keep_dropout)
    film_params = tf.contrib.layers.fully_connected(context,
                                       num_outputs=no_params * feature_size,
                                       activation_fn=None,
                                       reuse=reuse,
                                       scope="film_projection")

    film_params = tf.expand_dims(film_params, axis=[1])
    film_params = tf.tile(film_params, [1, no_elem, 1])


    if use_beta:
        gammas = film_params[:, :, :, :feature_size]
        betas = film_params[:, :, :, feature_size:]
        output = (1 + gammas) * features + betas
    else:
        gammas = film_params
        if use_softmax:
            gammas = tf.nn.softmax(gammas)
        output = (1 + gammas) * features

    return output



if __name__ == "__main__":

    ft = tf.placeholder(dtype=tf.float32, shape=[None, None, 256])
    ctx = tf.placeholder(dtype=tf.float32, shape=[None, 256, 49])

    gated_features1 = gated_attention(ft,ctx)
    gated_features2 = gated_film(ft,ctx)
