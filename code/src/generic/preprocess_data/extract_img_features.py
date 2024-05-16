#!/usr/bin/env python
import numpy
import os
import tensorflow as tf
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import h5py

from generic.data_provider.nlp_utils import DummyTokenizer
from generic.data_provider.iterator import Iterator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"

def extract_features(
        img_input,
        ft_output,
        network_ckpt, 
        dataset_cstor,
        dataset_args,
        batchifier_cstor,
        out_dir,
        set_type,
        batch_size,
        no_threads,
        gpu_ratio):

    # CPU/GPU option
    cpu_pool = Pool(no_threads, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)

    no_thread = 4
    gpu_ratio = 1.0
    # CPU/GPU option
    cpu_pool = Pool(no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)

    cfg = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
    cfg.log_device_placement=True
    cfg.gpu_options.allow_growth = True

    with tf.Session(config=cfg) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, network_ckpt)
    
        for one_set in set_type:
    
            print("Load dataset -> set: {}".format(one_set))
            dataset_args["which_set"] = one_set
            dataset = dataset_cstor(**dataset_args)
    
            # hack dataset to only keep one game by image
            image_id_set = {}
            games = []
            for game in dataset.games:
                if game.image.id not in image_id_set:
                    games.append(game)
                    image_id_set[game.image.id] = 1

            dataset.games = games
            no_images = len(games)
    
            source_name = os.path.basename(img_input.name[:-2])
            dummy_tokenizer = DummyTokenizer()
            batchifier = batchifier_cstor(tokenizer=dummy_tokenizer, sources=[source_name])
            iterator = Iterator(dataset,
                                batch_size=batch_size,
                                pool=cpu_pool,
                                batchifier=batchifier)
 
            ############################
            #  CREATE FEATURES
            ############################
            print("Start computing image features...")
            filepath = os.path.join(out_dir, "{}_features.h5".format(one_set))
            with h5py.File(filepath, 'w') as f:

                ft_shape = [int(dim) for dim in ft_output.get_shape()[1:]]
                #ft_shape = [100352]
                ft_dataset = f.create_dataset('features', shape=[no_images] + ft_shape, dtype=np.float32)
                idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
                pt_hd5 = 0
    
                for batch in tqdm(iterator):
                    feat = sess.run(ft_output, feed_dict={img_input: numpy.array(batch[source_name])})
                    
                    #print("ft_output:{}".format(np.shape(feat)))
                    """
                    ft_output:(64, 14, 14, 512)
                    ft_output:(64, 7, 7, 512)
                    """
                    # Store dataset
                    batch_size = len(batch["raw"])
                    #feat = tf.reshape(feat,shape=[batch_size,14*14*512])
                    #print("ft_output reshape:{}".format(np.shape(feat)))
                    """
                    ft_output reshape:(64, 100352)
                    ft_output reshape:(64, 25088)
                    """
                    ft_dataset[pt_hd5: pt_hd5 + batch_size] = feat
    
                    # Store idx to image.id
                    for i, game in enumerate(batch["raw"]):
                        idx2img[pt_hd5 + i] = game.image.id
    
                    # update hd5 pointer
                    pt_hd5 += batch_size
                    #if pt_hd5 >= 200:break 
                print("Start dumping file: {}".format(filepath))
            print("Finished dumping file: {}".format(filepath))
    
    print("Done!")

