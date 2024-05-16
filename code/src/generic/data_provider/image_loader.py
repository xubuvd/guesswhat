import os
from PIL import Image
import numpy as np
import h5py,h5pickle

from generic.data_provider.image_preprocessors import resize_image, scaled_crop_and_pad

# Why doing an image builder/loader?
# Well, there are two reasons:
#  - first you want to abstract the kind of image you are using (raw/conv/feature) when you are loading the dataset/batch.
#  One may just want... to load an image!
#  - One must optimize when to load the image for multiprocessing.
#       You do not want to serialize a 2Go of fc8 features when you create a process
#       You do not want to load 50Go of images at start
#
#   The Builder enables to abstract the kind of image you want to load. It will be used while loading the dataset.
#   The Loader enables to load/process the image when you need it. It will be used when you create the batch
#
#   Enjoy design patterns, it may **this** page of code complex but the it makes the whole project easier! Act Local, Think Global :P
#


class AbstractImgBuilder(object):
    def __init__(self, img_dir, is_raw, require_process=False):
        self.img_dir = img_dir
        self.is_raw = is_raw
        self.require_process = require_process

    def build(self, image_id, filename, **kwargs):
        return self

    def is_raw_image(self):
        return self.is_raw

    def require_multiprocess(self):
        return self.require_process

class AbstractImgLoader(object):
    def __init__(self, img_path):
        self.img_path = img_path

    def get_image(self, **kwargs):
        pass

class DummyImgBuilder(AbstractImgBuilder, AbstractImgLoader):
    def __init__(self, img_dir, size=1000):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
        self.size = size

    def build(self, image_id, filename, **kwargs):
        return self

    def get_image(self, **kwargs):
        return np.zeros(self.size)

class ErrorImgLoader(AbstractImgLoader):
    def __init__(self, img_path):
        AbstractImgLoader.__init__(self, img_path)

    def get_image(self, **kwargs):
        assert False, "The image/crop is not available in file: {}".format(self.img_path)


h5_basename="features.h5"
h5_feature_key="features"
h5_idx_key="idx2img"

class h5FeatureBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, bufferize):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
        self.bufferize = bufferize
        self.h5files = dict()
        self.img2idx = dict()

    def build(self, image_id, filename, optional=True, which_set=None,**kwargs):

        # Is the h5 features split into train/val/etc. files or gather into a single file
        if which_set is not None:
            h5filename = which_set + "_" + h5_basename
        else:
            h5filename = h5_basename

        # Build full bath
        h5filepath = os.path.join(self.img_dir,h5filename)

        # Retrieve
        if h5filename not in self.h5files:
            # Load file pointer to h5
            #h5file = h5py.File(h5filepath, 'r')
            """bug fixed with https://github.com/Exteris/h5pickle"""
            h5file = h5pickle.File(h5filepath, 'r',skip_cache=True)

            # hd5 requires continuous id while image_id can be very diverse.
            # We then need a mapping between both of them
            if h5_idx_key in h5file:
                # Retrieve id mapping from file
                img2idx = {id_img : id_h5 for id_h5, id_img in enumerate(h5file[h5_idx_key])}
            else:
                # Assume their is a perfect identity between image_id and h5_id
                no_images = h5file[h5_feature_key].shape[0]
                img2idx = {k : k for k in range(no_images) }

            self.h5files[h5filename] = h5file
            self.img2idx[h5filename] = img2idx
        else:
            h5file = self.h5files[h5filename]
            img2idx = self.img2idx[h5filename]

        if self.bufferize:
            if (optional and image_id in img2idx) or (not optional):
                return h5FeatureBufloader(h5filepath, h5file=h5file, id=img2idx[image_id])
            else:
                return ErrorImgLoader(h5filepath)
        else:
            #data/vgg16_pool5/ft_vgg_img/train_features.h5   image_id:240684
            is_exist = False
            if image_id in img2idx:is_exist = True
            #if is_exist:
            return h5FeatureLoader(h5filepath, h5file=h5file, id=img2idx[image_id],is_exist=is_exist)
            #else:
            #    print("{}\t{}\tnot in image2idx".format(h5filepath,image_id))
            #    return h5FeatureLoader(h5filepath, h5file=h5file, id=0,is_exist=is_exist)

# Load while creating batch
class h5FeatureLoader(AbstractImgLoader):
    def __init__(self, img_path, h5file, id,is_exist):
        AbstractImgLoader.__init__(self, img_path)
        self.h5file = h5file
        self.id = id
        self.is_exist = is_exist

    def get_image(self, **kwargs):
        return self.h5file[h5_feature_key][self.id]#(7,7,512), (36,2048)
        #if self.is_exist:
        #    return self.h5file[h5_feature_key][self.id]#(7,7,512)
        #else:
        #    return np.zeros((7,7,512))

# Load while loading dataset (requires a lot of memory)
class h5FeatureBufloader(AbstractImgLoader):
    def __init__(self, img_path, h5file, id):
        AbstractImgLoader.__init__(self, img_path)
        self.data = h5file[h5_feature_key][id]

    def get_image(self, **kwargs):
        return self.data

class RawImageBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, width, height, channel=None):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=True, require_process=True)
        self.width = width
        self.height = height
        self.channel = channel

    def build(self, image_id, filename, **kwargs):
        img_path = os.path.join(self.img_dir, filename)
        return RawImageLoader(img_path, self.width, self.height, channel=self.channel)

class RawImageLoader(AbstractImgLoader):
    def __init__(self, img_path, width, height, channel):
        AbstractImgLoader.__init__(self, img_path)
        self.width = width
        self.height = height
        self.channel = channel


    def get_image(self, **kwargs):
        img = Image.open(self.img_path).convert('RGB')

        img = resize_image(img, self.width , self.height)
        img = np.array(img, dtype=np.float32)

        if self.channel is not None:
            img -= self.channel[None, None, :]

        return img

class RawCropBuilder(AbstractImgBuilder):
    def __init__(self, data_dir, width, height, scale, channel=None):
        AbstractImgBuilder.__init__(self, data_dir, is_raw=True, require_process=True)
        self.width = width
        self.height = height
        self.channel = channel
        self.scale = scale

    def build(self, object_id, filename, **kwargs):
        bbox = kwargs["bbox"]
        img_path = os.path.join(self.img_dir, filename)
        return RawCropLoader(img_path, self.width, self.height, scale=self.scale, bbox=bbox, channel=self.channel)

class RawCropLoader(AbstractImgLoader):
    def __init__(self, img_path, width, height, scale, bbox, channel):
        AbstractImgLoader.__init__(self, img_path)
        self.width = width
        self.height = height
        self.channel = channel
        self.bbox = bbox
        self.scale = scale

    def get_image(self, **kwargs):

        img = Image.open(self.img_path).convert('RGB')

        crop = scaled_crop_and_pad(raw_img=img, bbox=self.bbox, scale=self.scale)
        crop = resize_image(crop, self.width , self.height)
        crop = np.array(crop, dtype=np.float32)

        if self.channel is not None:
            crop -= self.channel[None, None, :]

        return crop

def get_img_builder(config, image_dir, is_crop=False, bufferize=None):

    image_input = config["image_input"]
    if image_input in ["fc8","fc7"]:
        bufferize = bufferize if bufferize is not None else True
        loader = h5FeatureBuilder(image_dir, bufferize=bufferize)
    elif image_input in ["conv", "raw_h5","conv5/conv5_3","pool5"]:
        bufferize = bufferize if bufferize is not None else False
        loader = h5FeatureBuilder(image_dir, bufferize=bufferize)
    elif image_input in ["faster-rcnn"]:
        bufferize = bufferize if bufferize is not None else True
        loader = h5FeatureBuilder(image_dir, bufferize=bufferize)
    elif image_input == "raw":
        if is_crop:
            loader = RawCropBuilder(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    scale=config["scale"],
                                    channel=config.get("channel", None))
        else:
            loader = RawImageBuilder(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    channel=config.get("channel", None))
    else:
        assert False, "incorrect image input: {}".format(image_input)

    return loader




# Legacy code
#---------------------------------------------
#
# class ConvBuilder(AbstractImgBuilder):
#     def __init__(self, img_dir):
#         AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
#
#     def build(self, image_id, filename, **kwargs):
#         img_path = os.path.join(self.img_dir, "{}.npz".format(image_id))
#         return ConvLoader(img_path)
#
# class ConvLoader(AbstractImgLoader):
#     def __init__(self, img_path):
#         AbstractImgLoader.__init__(self, img_path)
#
#     def get_image(self, **kwargs):
#         return np.load(self.img_path)['x']
# class fcBuilder(AbstractImgBuilder):
#     def __init__(self, img_path):
#         AbstractImgBuilder.__init__(self, img_path, is_raw=False)
#         self.fc8 = pickle_loader(img_path)
#
#     # Trick to avoid serializing complete fc8 dictionary.
#     # We wrap the fc8 into a separate object which does not contain
#     # the *full* fc8 dictionary.
#
#     def build(self, image_id, filename, **kwargs):
#         """
#         :param image_id: (or object_id) id of the fc8 to load
#         :param filename: N/A
#         :param kwargs: optional -> do fc8 must be present in file
#         :return: fcLoader for the current id
#         """
#
#         one_fc8 = self.fc8.get(image_id, None)
#
#         assert one_fc8 is not None or kwargs.get("optional", True), \
#             "fc8 (id:{}) is missing in file: {}".format(image_id, self.img_dir)
#
#         return fcLoader(self.img_dir, one_fc8)
#
# class fcLoader(AbstractImgLoader):
#     def __init__(self,data_dir, fc8):
#         AbstractImgLoader.__init__(self, data_dir)
#         self.fc8 = fc8
#
#     def get_image(self, **kwargs):
#         assert self.fc8 is not None
#         return self.fc8
