import os
import pickle

import tensorflow as tf

def create_resnet_saver(networks):

    if not isinstance(networks, list):
        networks = [networks]

    resnet_vars = dict()
    for network in networks:

        start = len(network.scope_name) + 1
        for v in network.get_resnet_parameters():
            resnet_vars[v.name[start:-2]] = v

    return tf.train.Saver(resnet_vars)

def load_checkpoint(sess, saver, args, save_path,default=0):
    ckpt_path = save_path.format('params.ckpt')

    if args.continue_exp:
        if not os.path.exists(save_path.format('checkpoint')):
            raise ValueError("Checkpoint " + save_path.format('checkpoint') + " could not be found.")

        saver.restore(sess, ckpt_path)
        status_path = save_path.format('status.pkl')
        status = pickle.load(open(status_path, 'rb'))

        return status['epoch'] + 1,status['best_val_loss']

    if args.load_checkpoint is not None:
        #if not os.path.exists(save_path.format('checkpoint')):
        #    raise ValueError("Checkpoint " + args.load_checkpoint + " could not be found.")
        saver.restore(sess, args.load_checkpoint)

        return 0,default

    return 0,default

