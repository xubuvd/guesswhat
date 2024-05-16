import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.layers as tfc_layers

def create_optimizer(network, config, finetune=list(), optim_cst=tf.train.AdamOptimizer, var_list=None, apply_update_ops=True, loss=None, staircase=False,global_step=None,momentum=False):

    # Retrieve conf
    lrt = config['optimizer']['learning_rate']
    decay_epoches = config['optimizer'].get('decay_epoches',1)
    decay_steps = config['optimizer']['decay_steps']*decay_epoches
    decay_rate = config['optimizer']['decay_rate']
    clip_val = config['optimizer'].get('clip_val', 0.)
    weight_decay = config['optimizer'].get('weight_decay', 0.)

    # create optimizer
    if global_step is not None:
        learning_rate = tf.train.exponential_decay(learning_rate=lrt, global_step=global_step,decay_steps=decay_steps,decay_rate=decay_rate,staircase=staircase)
    else:
        learning_rate = lrt
 
    if momentum:
        #momentum_exp = tf.train.exponential_decay(learning_rate=0.80, global_step=global_step,decay_steps=config['optimizer']['decay_steps']*25,decay_rate=1.035,staircase=True)
        optimizer = optim_cst(learning_rate=learning_rate,momentum=0.9)
    else:
        #momentum_exp = 0.5
        optimizer = optim_cst(learning_rate=learning_rate)
 
    # Extract trainable variables if not provided
    if var_list is None:
        var_list = network.get_parameters(finetune=finetune)

    # Apply weight decay
    if loss is None:
        loss = network.get_loss()

    # Apply weight decay
    training_loss = loss
    if weight_decay > 0:
        training_loss = loss + l2_regularization(var_list, weight_decay=weight_decay)

    # compute gradient
    grad = optimizer.compute_gradients(training_loss, var_list=var_list)

    # apply gradient clipping
    if clip_val > 0:
        grad = clip_gradient(grad, clip_val=clip_val)

    # Apply gradients
    if apply_update_ops:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if global_step is not None:
                optimize = optimizer.apply_gradients(grad,global_step=global_step)
            else:
                optimize = optimizer.apply_gradients(grad)
    else:
        if global_step is not None:
            optimize = optimizer.apply_gradients(grad,global_step=global_step)
        else:
            optimize = optimizer.apply_gradients(grad)

    accuracy = network.get_accuracy()
    return optimize, [loss, accuracy],learning_rate


def create_multi_gpu_optimizer(networks, config, finetune=list(), optim_cst=tf.train.AdamOptimizer):

    # Retrieve conf
    lrt = config['learning_rate']
    clip_val = config.get('clip_val', 0.)
    weight_decay = config.get('weight_decay', 0.)
    weight_decay_remove = config.get('weight_decay_remove', [])

    # Create optimizer
    optimizer = optim_cst(learning_rate=lrt)

    gradients, losses, accuracies = [], [], []
    for i, network in enumerate(networks):
        with tf.device('gpu:{}'.format(i)):

            # Retrieve trainable variables from network
            train_vars = network.get_parameters(finetune=finetune)

            # Apply weight decay
            loss = network.get_loss()

            training_loss = loss
            if weight_decay > 0:
                training_loss += l2_regularization(train_vars, weight_decay=weight_decay, weight_decay_remove=weight_decay_remove)

            # compute gradient
            grads = optimizer.compute_gradients(training_loss, train_vars)
            gradients.append(grads)

            # Retrieve training loss
            losses.append(network.get_loss())
            # Retrieve evaluation loss
            accuracies.append(network.get_accuracy())
    # Synchronize and average gradient/loss/accuracy
    avg_grad = average_gradient(gradients)
    avg_loss = tf.reduce_mean(tf.stack(losses))
    avg_accuracy = tf.reduce_mean(tf.stack(accuracies))
    # Clip gradient
    if clip_val > 0:
        avg_grad = clip_gradient(avg_grad, clip_val=clip_val)
    # Apply gradients
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize = optimizer.apply_gradients(avg_grad)
    return optimize, [avg_loss, avg_accuracy]

def clip_gradient(gvs, clip_val):
    clipped_gvs = list()
    for grad, var in gvs:
        #if grad is not None:
        cgrad = tf.clip_by_norm(grad, clip_val)
        #else:
        #    print("ValueError: None values not supported - grad:{}\tvar:{}".format(grad,var))
        clipped_gvs.append((cgrad,var))
    #clipped_gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]
    return clipped_gvs

def l2_regularization(params, weight_decay, weight_decay_remove=list()):
    with tf.variable_scope("l2_normalization"):
        params = [v for v in params if
                      not any([(needle in v.name) for needle in weight_decay_remove])]
        regularizer = tfc_layers.l2_regularizer(scale=weight_decay)

        return tfc_layers.apply_regularization(regularizer, weights_list=params)

def average_gradient(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

