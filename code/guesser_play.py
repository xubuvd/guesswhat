import argparse
import logging
import os
import numpy as np
from multiprocessing import Pool
from distutils.util import strtobool
import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint, create_resnet_saver
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.guesser.guesser_network import GuesserNetwork

os.environ["CUDA_VISIBLE_DEVICES"] = str("1")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Guesser network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-config", type=str, help="Configuration file")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp",type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
    parser.add_argument("-guesser_identifier", type=str, required=True, help='Guesser identifier')

    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir,identifier=args.guesser_identifier)
    logger = logging.getLogger()
    logger.info('guesser_identifier:{}'.format(args.guesser_identifier))

    ###############################
    #  LOAD DATA
    #############################
    image_builder, crop_builder = None, None
    # Load image
    logger.info('Loading images..')
    use_resnet = False
    if 'image' in config['model']:
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()
        assert False, "Guesser + Image is not yet available"

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    # Build Network
    logger.info('Building network..')
    network = GuesserNetwork(config['model'], num_words=tokenizer.no_words)

    # Build Optimizer
    _,outputs,_ = create_optimizer(network, config)

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder)

    ###############################
    #  START  TRAINING
    #trainset############################

    # Load config
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    # create a saver to store/load checkpoint
    guesser_var = [v for v in tf.global_variables() if "guesser" in v.name and "Adam" not in v.name]
    saver = tf.train.Saver(var_list=guesser_var)

    # Retrieve only resnet variabes
    if use_resnet:resnet_saver = create_resnet_saver([network])

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    cfg = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    cfg.log_device_placement=True
    cfg.gpu_options.allow_growth = True

    with tf.Session(config=cfg) as sess:
        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))
        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))

        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = QuestionerBatchifier(tokenizer, sources, status=('success',))
        
        train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True,augmented=False,aug_factor=1)
        train_loss, train_accuracy = evaluator.process(sess, train_iterator, outputs=outputs)

        valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size,
                                      batchifier=batchifier,
                                      shuffle=False)
        valid_loss, valid_accuracy = evaluator.process(sess, valid_iterator, outputs=outputs)

        logger.info("Training loss: {}".format(train_loss))
        logger.info("Training accuracy: {}".format(train_accuracy))
        logger.info("Training error: {}".format(1-train_accuracy))
        logger.info("Validation loss: {}".format(valid_loss))
        logger.info("Validation accuracy: {}".format(valid_accuracy))
        logger.info("Validation error: {}".format(1-valid_accuracy))
        
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=True)
        [test_loss, test_accuracy] = evaluator.process(sess, test_iterator,outputs)

        logger.info("Testing loss: {}".format(test_loss))
        logger.info("Testing accuracy: {}".format(test_accuracy))
        logger.info("Testing error: {}".format(1 - test_accuracy))

