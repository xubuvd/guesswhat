import argparse
import logging
import os
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

from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.oracle.oracle_network import OracleNetwork

os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Oracle network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-crop_dir", type=str, help='Directory with images')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

    args = parser.parse_args()

    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    logger = logging.getLogger()

    #copy code
    code_path = save_path.format('')
    #if not os.path.exists(code_path):os.mkdir(code_path)
    cmd="cp -rf /home/pw/guesswhat/src {}".format(code_path)
    os.system(cmd)
    cmd="cp -rf /home/pw/guesswhat/Train_Oracle.sh {}".format(code_path)
    os.system(cmd)

    # Load config
    resnet_version = config['model']["image"].get('resnet_version', 50)
    finetune = config["model"]["image"].get('finetune', list())
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    image_builder, crop_builder = None, None
    use_resnet = False
    if config['inputs'].get('image', False):
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

    if config['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(config['model']['crop'], args.crop_dir, is_crop=True)
        use_resnet = crop_builder.is_raw_image()

    # Load data
    logger.info('Loading data..')
    trainset = OracleDataset.load(args.data_dir, "train", image_builder, crop_builder)
    validset = OracleDataset.load(args.data_dir, "valid", image_builder, crop_builder)
    testset = OracleDataset.load(args.data_dir, "test", image_builder, crop_builder)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    # Build Network
    logger.info('Building network..')
    network = OracleNetwork(config, num_words=tokenizer.no_words)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer,outputs,_ = create_optimizer(network, config, finetune=finetune)

    ###############################
    #  START  TRAINING
    #############################

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()
    resnet_saver = None

    # Retrieve only resnet variabes
    if use_resnet:
        resnet_saver = create_resnet_saver([network])

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
    cfg = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    cfg.log_device_placement=True
    cfg.gpu_options.allow_growth = True

    with tf.Session(config=cfg) as sess:
        #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))
        #Sources: is_training, question, seq_length, category, spatial, answer

        sess.run(tf.global_variables_initializer())
        if use_resnet:
            resnet_saver.restore(sess, os.path.join(args.data_dir, 'resnet_v1_{}.ckpt'.format(resnet_version)))
        start_epoch,best_val_err = load_checkpoint(sess, saver, args, save_path,default=0)

        #best_val_err = 0
        best_train_err = None

        print("start_epoch:{}".format(start_epoch))
        # create training tools
        evaluator = Evaluator(sources, network.scope_name)
        batchifier = OracleBatchifier(tokenizer, sources, status=config['status'])

        for t in range(start_epoch, no_epoch):
            logger.info('Epoch {}/{}..'.format(t + 1,no_epoch))

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            train_loss, train_accuracy = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False)
            valid_loss, valid_accuracy = evaluator.process(sess, valid_iterator, outputs=outputs)

            logger.info("Training loss:\t{}".format(train_loss))
            logger.info("Training accuracy:\t{}".format(train_accuracy))
            logger.info("Training error:\t{}".format(1.0-train_accuracy))
            logger.info("Validation loss:\t{}".format(valid_loss))
            logger.info("Validation accuracy:\t{}".format(valid_accuracy))
            logger.info("Validation error:\t{}".format(1.0-valid_accuracy))

            if valid_accuracy > best_val_err:
                best_train_err = train_accuracy
                best_val_err = valid_accuracy
                model_path = saver.save(sess, save_path.format('params.ckpt'))
                logger.info("Oracle checkpoint saved to {}".format(model_path))
                pickle_dump({'epoch': t,'best_val_loss':best_val_err}, save_path.format('status.pkl'))

        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size*2,
                                 batchifier=batchifier,
                                 shuffle=True)
        [test_loss, test_accuracy] = evaluator.process(sess, test_iterator, outputs)

        logger.info("Testing loss:\t{}".format(test_loss))
        logger.info("Testing accuracy:\t{}".format(test_accuracy))
        logger.info("Testing error:\t{}".format(1.0-test_accuracy))

