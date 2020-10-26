import json
import os
import six

import chainer
import numpy
from PIL import Image

from chainer.dataset import DatasetMixin
from chainer.datasets.image_dataset import _check_pillow_availability
from imgaug import augmenters as iaa
from imgaug import parameters as iap

from utils.datatypes import Size
from utils.img_utils import prepare_image


class BaseImageDataset(DatasetMixin):

    def __init__(self, gt_file, image_size, root='.', dtype=None, transform_probability=0, image_mode='L', keep_aspect_ratio=False, resize_after_load=True):
        _check_pillow_availability()
        assert isinstance(gt_file, six.string_types), "paths must be a file name!"
        assert os.path.splitext(gt_file)[-1] == ".json", "You have to supply gt information as json file!"

        if not isinstance(image_size, Size):
            image_size = Size(*image_size)

        with open(gt_file) as handle:
            self.gt_data = json.load(handle)

        self.root = root
        self.dtype = chainer.get_dtype(dtype)
        self.image_size = image_size
        self.transform_probability = transform_probability
        self.keep_aspect_ratio = keep_aspect_ratio
        self.resize_after_load = resize_after_load
        self.image_mode = image_mode
        self.augmentations = self.init_augmentations()

    def init_augmentations(self):
        if self.transform_probability > 0:
            augmentations = iaa.Sometimes(
                self.transform_probability,
                iaa.Sequential([
                    iaa.SomeOf(
                        (1, None),
                        [
                            iaa.AddToHueAndSaturation(iap.Uniform(-20, 20), per_channel=True),
                            iaa.GaussianBlur(sigma=(0, 1.0)),
                            iaa.LinearContrast((0.75, 1.0)),
                            iaa.PiecewiseAffine(scale=(0.01, 0.02), mode='edge'),
                        ],
                        random_order=True
                    )
                ])
            )
        else:
            augmentations = None
        return augmentations

    def maybe_augment(self, image):
        if self.augmentations is not None:
            image_data = numpy.asarray(image)
            image_data = self.augmentations.augment_image(image_data)
            image = Image.fromarray(image_data)

        return image

    def __len__(self):
        return len(self.gt_data)

    def get_example(self, i):
        gt_data = self.gt_data[i]
        image = self.load_image(gt_data['file_name'])
        has_text = numpy.array(gt_data['has_handwriting'], dtype='int32')

        return {
            "image": image,
            "has_text": has_text,
        }

    def load_image(self, file_name, with_augmentation=True):
        with Image.open(os.path.join(self.root, file_name)) as the_image:
            the_image = the_image.convert(self.image_mode).convert("RGB")
            if with_augmentation:
                the_image = self.maybe_augment(the_image)
            image = prepare_image(
                the_image,
                self.image_size,
                numpy,
                self.keep_aspect_ratio,
                do_resize=self.resize_after_load,
            )
        return image
