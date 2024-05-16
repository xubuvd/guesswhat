#!/usr/bin/env python
import os
from tqdm import tqdm
import numpy as np

import h5py

from multiprocessing import Pool

from generic.data_provider.iterator import Iterator
from generic.data_provider.nlp_utils import DummyTokenizer


def extract_raw(
        image_shape,
        dataset_cstor,
        dataset_args,
        batchifier_cstor,
        source_name,
        out_dir,
        set_type,
        no_threads,
):

    for one_set in set_type:

        ############################
        #   LOAD DATASET
        ############################

        print("Load dataset...")
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

        # prepare batch builder
        dummy_tokenizer = DummyTokenizer()
        batchifier = batchifier_cstor(tokenizer=dummy_tokenizer, sources=[source_name])
        cpu_pool = Pool(no_threads, maxtasksperchild=1000)
        iterator = Iterator(dataset,
                            batch_size=64,
                            pool=cpu_pool,
                            batchifier=batchifier)

        filepath = os.path.join(out_dir, "{}_features.h5".format(one_set))
        with h5py.File(filepath, 'w') as f:

            feat_dataset = f.create_dataset('features', shape=[no_images] + image_shape, dtype=np.float32)
            idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
            pt_hd5 = 0

            for batch in tqdm(iterator):

                # Store dataset
                batch_size = len(batch["raw"])
                feat_dataset[pt_hd5: pt_hd5 + batch_size] = batch[source_name]

                # Store idx to image.id
                for i, game in enumerate(batch["raw"]):
                    idx2img[pt_hd5 + i] = game.image.id

                # update hd5 pointer
                pt_hd5 += batch_size

            print("Start dumping file: {}".format(filepath))
        print("Finished dumping file: {}".format(filepath))

    print("Done!")
