import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import UniformUnitScaling, Constant
from neural_toolbox import utils

def compute_attention(feature_maps, context, no_mlp_units, reuse=False):
    with tf.variable_scope("attention"):

        if len(feature_maps.get_shape()) == 3:
            h = tf.shape(feature_maps)[1]  # when the shape is dynamic (attention over lstm)
            w = 1
            c = int(feature_maps.get_shape()[2])
        else:
            h = int(feature_maps.get_shape()[1])
            w = int(feature_maps.get_shape()[2])
            c = int(feature_maps.get_shape()[3])

        s = int(context.get_shape()[1])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h * w, c])

        context = tf.expand_dims(context, axis=1)
        context = tf.tile(context, [1, h * w, 1])

        embedding = tf.concat([feature_maps, context], axis=2)
        embedding = tf.reshape(embedding, shape=[-1, s + c])

        # compute the evidence from the embedding
        with tf.variable_scope("mlp"):
            e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu", reuse=reuse)
            e = utils.fully_connected(e, 1, scope='out', reuse=reuse)

        e = tf.reshape(e, shape=[-1, h * w, 1])

        # compute the softmax over the evidence
        alpha = tf.nn.softmax(e, dim=1)

        # apply soft attention
        soft_attention = feature_maps * alpha
        soft_attention = tf.reduce_sum(soft_attention, axis=1)

    return soft_attention


# cf https://arxiv.org/abs/1610.04325
def compute_glimpse(feature_maps, context, no_glimpse, glimpse_embedding_size, keep_dropout, reuse=False):
    with tf.variable_scope("glimpse"):
        h = int(feature_maps.get_shape()[1])
        w = int(feature_maps.get_shape()[2])
        c = int(feature_maps.get_shape()[3])

        # reshape state to perform batch operation
        context = tf.nn.dropout(context, keep_dropout)
        projected_context = utils.fully_connected(context, glimpse_embedding_size,
                                                  scope='hidden_layer', activation="tanh",
                                                  use_bias=False, reuse=reuse)

        projected_context = tf.expand_dims(projected_context, axis=1)
        projected_context = tf.tile(projected_context, [1, h * w, 1])
        projected_context = tf.reshape(projected_context, [-1, glimpse_embedding_size])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h * w, c])

        glimpses = []
        with tf.variable_scope("glimpse"):
            g_feature_maps = tf.reshape(feature_maps, shape=[-1, c])  # linearise the feature map as as single batch
            g_feature_maps = tf.nn.dropout(g_feature_maps, keep_dropout)
            g_feature_maps = utils.fully_connected(g_feature_maps, glimpse_embedding_size, scope='image_projection',
                                                   activation="tanh", use_bias=False, reuse=reuse)

            hadamard = g_feature_maps * projected_context
            hadamard = tf.nn.dropout(hadamard, keep_dropout)

            e = utils.fully_connected(hadamard, no_glimpse, scope='hadamard_projection', reuse=reuse)
            e = tf.reshape(e, shape=[-1, h * w, no_glimpse])

            for i in range(no_glimpse):
                ev = e[:, :, i]
                alpha = tf.nn.softmax(ev)
                # apply soft attention
                soft_glimpses = feature_maps * tf.expand_dims(alpha, -1)
                soft_glimpses = tf.reduce_sum(soft_glimpses, axis=1)

                glimpses.append(soft_glimpses)

        full_glimpse = tf.concat(glimpses, axis=1)

    return full_glimpse

def Normalized_Objects(x,pi,reuse=False,is_train=True):
    """
    normalize the initial objects under the distribution of pi
    args:
        x, the initial objects set, [batch_size, objects_num, object_feature_size]
        pi, the distribution, [batch_size, objects_num]
    returns:
        norm_x, the normalized objects, [batch_size, objects_num, object_feature_size]
        norm_x_re, the flatten objects, [batch_size, objects_num*object_feature_size]
    """
    batch_size = x.get_shape()[0].value
    objects_num = x.get_shape()[1].value
    object_feature_size = x.get_shape()[2].value

    #l2 normlize along with the column
    #norm_x = tf.nn.l2_normalize(tf.multiply(x,tf.expand_dims(pi,-1)), axis=1)#(3,4,5)
    #norm_x = tf.contrib.layers.batch_norm(tf.multiply(x,tf.expand_dims(pi,-1),decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,is_training=True,scope="batch_norm")

    #skip-connection followed by BatchNorm
    #norm_x = tf.contrib.layers.batch_norm(tf.multiply(x,tf.expand_dims(pi,-1)) + x,decay=0.9,\
    #        updates_collections=None,epsilon=1e-5,scale=True,fused=True,data_format='NCHW',is_training=is_train,scope="BatchNormObjects",reuse=reuse)
    #norm_x_flatten = tf.reshape(norm_x, shape=[batch_size,objects_num*object_feature_size])
    norm_x = tf.multiply(x,tf.expand_dims(pi,-1))

    return norm_x

def SelfDifferenceAttention(x,pi,h=1,g=2,scope_name="Self_Difference_Attention",reuse=False,is_train=True,weight_initializer=tf.contrib.layers.xavier_initializer()):
    """
    compute self-difference attention over the current objects set that is normlized already.
    args:
        norm_x, the normalized current objects set, [batch_size, objects_num, object_feature_size], comes from Normalized_Objects()
        h, the number of multi-head
        g, glimpses
    returns:
        v, the salient visual feature for generating a new question, [batch_size, object_feature_size]
    For example,
        norm_x = (3,4,5)
        h = 2,
        g = 2
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        
        batch_size = x.get_shape()[0].value
        objects_num = x.get_shape()[1].value
        object_feature_size = x.get_shape()[2].value
        
        norm_x = tf.multiply(x,tf.expand_dims(pi,-1))

        norm_x_flatten = tf.reshape(norm_x, shape=[batch_size,objects_num*object_feature_size])   
        #(batch_size, objects_num, objects_num*object_feature_size)
        res = tf.multiply(tf.tile(norm_x,[1,1,objects_num]),tf.subtract(tf.tile(norm_x,[1,1,objects_num]),tf.expand_dims(norm_x_flatten,axis=1)))#(3,4,20)
        out_v = list()
        for head in range(h):
            reuse_t = True
            if head == 0: reuse_t = reuse
            #(md,g)=(4*5,2)
            W = tf.get_variable("WH_{}".format(head),[objects_num*object_feature_size,g],initializer=weight_initializer)
            b = tf.get_variable("bH_{}".format(head), [g,],initializer=tf.zeros_initializer())

            r1 = tf.einsum('ijk,kl->ijl',res,W) + b#(3,4,20)*(20,2) = (3,4,2)
            p1 = tf.transpose(tf.nn.softmax(r1),[0,2,1])#(3,4,2) -> (3,2,4) data_format='NCHW'
            p2 = tf.nn.selu(tf.matmul(p1, norm_x))#(3,2,4)*(3,4,5) = (3,2,5)
            #v = tf.nn.swish(tf.reshape(p2,[batch_size,-1]))#(3,2,5) -> (3,2*5)
            v = tf.reshape(p2,[batch_size,-1])#(3,2,5) -> (3,2*5)
            out_v.append(v)
        #out_v, list of size h.
        v_cat = tf.concat(out_v,axis=1)#(3, 2*5*h)    
        Wo = tf.get_variable("WH_Output",[g*object_feature_size*h,object_feature_size],initializer=weight_initializer)
        bo = tf.get_variable("bH_Output", [object_feature_size,],initializer=tf.zeros_initializer())
        v_output = tf.nn.selu(tf.matmul(v_cat,Wo) + bo)#(3, 5)
        return v_output

def SelfDifferenceAttention2(norm_x,h=1,g=2,scope_name="Self_Difference_Attention",reuse=False,is_train=True,weight_initializer=tf.contrib.layers.xavier_initializer()):
    """
    the same as SelfDifferenceAttention()
    """
    with tf.variable_scope(scope_name, reuse=reuse):
 
        batch_size = norm_x.get_shape()[0].value
        objects_num = norm_x.get_shape()[1].value
        object_feature_size = norm_x.get_shape()[2].value
        norm_x_re = tf.reshape(norm_x, shape=[batch_size,objects_num*object_feature_size])
        U = tf.get_variable("W_U_left",[object_feature_size,object_feature_size],initializer=weight_initializer)
        #U*o_i
        Ux = tf.einsum('ijk,kl->ijl',norm_x,U)#(batch_size, objects_num, object_feature_size)
        xnorm_ex = tf.tile(Ux,[1,1,objects_num])#(batch_size,objects_num*object_feature_size)

        #(o_i - o_j)
        subObjs = tf.reshape(tf.subtract(tf.tile(norm_x,[1,1,objects_num]),tf.expand_dims(norm_x_re,axis=1)),\
            shape=[batch_size,objects_num,objects_num,object_feature_size])#(batch_size, objects_num,objects_num, object_feature_size)
        V = tf.get_variable("W_V_right",[object_feature_size,object_feature_size],initializer=weight_initializer)
        VObjs = tf.einsum('ijkl,ln->ijkn',subObjs,V)#(batch_size, objects_num,objects_num, object_feature_size)

        res = tf.reshape(tf.multiply(tf.tile(tf.expand_dims(Ux,axis=2),[1,1,objects_num,1]), VObjs),shape=[batch_size,objects_num,objects_num*object_feature_size])
        out_v = list()
        for head in range(h):
            reuse_t = True
            if head == 0: reuse_t = reuse
            W = tf.get_variable("WH_{}".format(head),[objects_num*object_feature_size,g],initializer=weight_initializer)
            b = tf.get_variable("bH_{}".format(head), [g,],initializer=tf.zeros_initializer())

            r1 = tf.einsum('ijk,kl->ijl',res,W) + b#(3,4,20)*(20,2) = (3,4,2)
            p1 = tf.transpose(tf.nn.softmax(r1),[0,2,1])#(3,4,2) -> (3,2,4) data_format='NCHW'
            p2 = tf.matmul(p1, norm_x)
            #tf.contrib.layers.batch_norm(tf.matmul(p1, norm_x),decay=0.9,updates_collections=None,fused=True,data_format='NCHW',\
            #epsilon=1e-5,scale=True,is_training=is_train,scope="BatchNormSelfDiffAttn",reuse=reuse_t)
            #p2: (3,2,4)*(3,4,5) = (3,2,5)
            v = tf.nn.swish(tf.reshape(p2,[batch_size,-1]))#(3,2,5) -> (3,2*5)
            out_v.append(v)
        #out_v, list of size h.
        v_cat= tf.concat(out_v,axis=1)#(3, 2*5*h)
        Wo = tf.random_normal(shape=[g*object_feature_size*h,object_feature_size],mean=0.0,stddev=1.0,dtype=tf.float32,seed=1337)
        v_output = tf.matmul(v_cat,Wo)#(3, 5)
        return v_output

def Decision_Making2(h, x, prev_p, scope_name="Decision_Making",reuse=False,weight_initializer=tf.contrib.layers.xavier_initializer()):
    """
    args:
        h, the meaning vector of question and answer, [batch_size, emb_size]
        current_objects, [batch_size, objects_num, object_feature_size]
        prev_p, the previous distribution over the objects set, [batch_size, objects_num]
    returns:
        prob, [batch_size, objects_num]
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        object_feature_size = x.get_shape()[2].value
        emb_size = h.get_shape()[1].value

        current_objects = tf.multiply(x,tf.expand_dims(prev_p,-1))

        U = tf.get_variable("Weight_Decision",[object_feature_size,emb_size],initializer=weight_initializer)
        #Hadamard(UO,h)
        z = tf.einsum('ijk,kl->ijl',current_objects,U)#(batch_size,objects_num, emb_size)
        r = tf.multiply(tf.expand_dims(h,axis=1),z)#(batch_size,objects_num, emb_size)
        p = tf.nn.softmax(tf.squeeze(tf.matmul(r, tf.expand_dims(h,axis=-1))))#(batch_size, objects_num)
        
        #accumulate product previous prob. over objects
        #p = tf.nn.l2_normalize(tf.multiply(p,prev_p),axis=1)#(batch_size, objects_num)
        accum_p = tf.multiply(p,prev_p)#(batch_size, objects_num)
        p_norm = tf.div(accum_p,tf.reduce_sum(accum_p, axis=1,keepdims=True))#(batch_size, objects_num)
        return p_norm

def compute_intrinsic_reward(prev_prob, curr_prob):
    """
    an intrinsic reward that comes from information gain between the questions.
    args:
        prev_prob, [batch_size, objects_num]
        curr_prob, [batch_size, objects_num]
    returns:
        rewards, [batch_size,]
    """
    epsilon = 1e-5
    prev_entropy = -np.sum(prev_prob*np.log2(prev_prob+epsilon), axis=1)
    curr_entropy = -np.sum(curr_prob*np.log2(curr_prob+epsilon), axis=1)
    return prev_entropy - curr_entropy

def compute_consin_distance(prev_h_Q, prev_h_Q_inv):
    """
    compute the consin distance between the two vectors
    args:
        prev_h_Q_inv, [batch_size, 1024]
        prev_h_Q,     [batch_size, 1024]
    returns:
        d, [1,batch_size], d in [0,-2]
    """

    x1 = tf.nn.l2_normalize(prev_h_Q, axis=1)#on row
    x2 = tf.nn.l2_normalize(prev_h_Q_inv, axis=1)#on row
    d = -tf.transpose(tf.losses.cosine_distance(labels=x1, predictions=x2, axis=1,reduction=tf.losses.Reduction.NONE),[1,0])#[1,batch_size]
    return d

