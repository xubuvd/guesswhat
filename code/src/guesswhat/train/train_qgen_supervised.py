import argparse
import logging
import os
import sys
import numpy as np
from multiprocessing import Pool
from distutils.util import strtobool

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#only want to use cpu version, set CUDA_VISIBLE_DEVICES to "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#or any {'0', '1', '2'}

if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('QGen network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=1.0, help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")
    parser.add_argument("-gpu", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES")

    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    logger = logging.getLogger()

    #copy code
    code_path = save_path.format('')
    #if not os.path.exists(code_path):os.mkdir(code_path)
    cmd="cp -rf /home/pw/guesswhat/src {}".format(code_path)
    os.system(cmd) 
    cmd="cp -rf /home/pw/guesswhat/Train_QGen.sh {}".format(code_path)
    os.system(cmd)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    #get_img_loader Build Network
    logger.info('Building network..')
    network = QGenNetworkLSTM(config["model"], num_words=tokenizer.no_words, policy_gradient=False)

    # Load image
    logger.info('Loading images..')
    image_loader = get_img_builder(config['model']['image'], args.img_dir)
    crop_loader = None#get_img_loader(config, 'crop', args.img_dir)
    
    # Build Optimizer
    logger.info('Building optimizer..')
    #decay_steps = int(0.2*trainset.n_examples()/batch_size)
    optimizer, outputs,_ = create_optimizer(network, config,staircase=True)

    ###############################
    #  LOAD DATA
    ###############################
    logger.info('Loading data..')
    used_for_training = config["model"]["used_num"] #None
    trainset = Dataset(folder=args.data_dir, which_set="train", image_builder=image_loader, crop_builder=crop_loader,used_num=used_for_training)
    validset = Dataset(folder=args.data_dir, which_set="valid", image_builder=image_loader, crop_builder=crop_loader,used_num=used_for_training)
    testset = Dataset(folder=args.data_dir, which_set="test", image_builder=image_loader, crop_builder=crop_loader,used_num=used_for_training)

    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    ###############################
    #  START TRAINING
    #############################

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list)

    # create a saver to store/load checkpoint
    #saver = tf.train.Saver()

    total_variable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    logger.info("total_variable_count:{}".format(total_variable_count))

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    cpu_num = int(os.environ.get('CPU_NUM', 10))
    #cfg = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,\
    #        device_count={"CPU": cpu_num},inter_op_parallelism_threads=cpu_num,intra_op_parallelism_threads=cpu_num)
    cfg = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
    cfg.log_device_placement=False
    cfg.gpu_options.allow_growth = False
    with tf.Session(config=cfg) as sess:
        
        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))
        
        sess.run(tf.global_variables_initializer())
        start_epoch,best_val_loss = load_checkpoint(sess, saver, args, save_path,default=1e5)

        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = QuestionerBatchifier(tokenizer, sources, status=('success',))

        #best_val_loss = 1e5
        for t in range(start_epoch, config['optimizer']['no_epoch']):

            logger.info('Epoch {}/{}..'.format(t,config['optimizer']['no_epoch']))

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            [train_loss, _] = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size,
                                      batchifier=batchifier,
                                      shuffle=False)
            [valid_loss, _] = evaluator.process(sess, valid_iterator, outputs=outputs)

            logger.info("Training loss: {}".format(train_loss))
            logger.info("Validation loss: {}".format(valid_loss))

            if valid_loss < best_val_loss:
                best_train_loss = train_loss
                best_val_loss = valid_loss
                modelpath = saver.save(sess, save_path.format('params.ckpt'))
                logger.info("Save checkpoint...")
                pickle_dump({'epoch': t,'best_val_loss':best_val_loss}, save_path.format('status.pkl'))
        #train_log.close()
        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=True)
        [test_loss, _] = evaluator.process(sess, test_iterator, outputs)
        logger.info("Testing loss: {}".format(test_loss))

