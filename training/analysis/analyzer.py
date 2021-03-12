import json
from pathlib import Path
from typing import Union, Tuple, List

import chainer
import numpy
from PIL import Image
from PIL.Image import Image as ImageClass
from chainer import Chain
from chainer.backend import get_array_module
from chainer.dataset import concat_examples

from datasets.prediction_dataset import PredictionDatasetMixin
from utils.backup import restore_backup

from utils.bbox.bbox import AxisAlignedBBox

Image.init()


def is_image(file_name: Path) -> bool:
    if not isinstance(file_name, Path):
        file_name = Path(file_name)
    return file_name.suffix.lower() in Image.EXTENSION.keys()


class Analyzer:

    def __init__(self, model_path: Path, device_id: int, needs_patches: bool = True):
        self.model_path = model_path
        self.device_id = device_id
        self.needs_patches = needs_patches

        log_file = self.model_path.parent / 'log'

        with log_file.open() as f:
            self.log_data = json.load(f)[0]

        self.prediction_helper = PredictionDatasetMixin(image_size=self.log_data['image_size'], max_size=2000)
        self.initialized = False
        self.network = None
        self.initialize()

    def initialize(self):
        if self.initialized:
            return

        self.initialized = True

        if self.device_id >= 0:
            with chainer.using_device(chainer.get_device(self.device_id)):
                self.network = self.load_network(self.device_id)
        else:
            self.network = self.load_network('@numpy')

    def load_network(self, device_id) -> Chain:
        net_class = restore_backup(self.log_data['net'], '.')
        net = net_class()

        with numpy.load(str(self.model_path)) as f:
            chainer.serializers.NpzDeserializer(f, strict=True).load(net)

        net = net.to_device(device_id)

        return net

    def evaluate_image(self, image: numpy.ndarray, return_boxes: bool = False) -> Union[bool, Tuple[bool, list]]:
        if self.needs_patches:
            patches, bboxes = self.prediction_helper.create_sliding_window(image)
        else:
            patches = image[numpy.newaxis, ...]

        network = self.network
        device = chainer.get_device(network.device)

        xp = numpy
        with chainer.using_device(device), chainer.configuration.using_config('train', False):
            predicted_patches = []
            for patch in patches:
                batch = [{'image': patch}]
                batch = concat_examples(batch, device)

                xp = get_array_module(batch['image'])
                predictions = network(**batch)
                predicted_patches.append(xp.argmax(predictions.array, axis=1))

            predicted_patches = xp.stack(predicted_patches, axis=0)
            contains_handwriting = chainer.backends.cuda.to_cpu(predicted_patches == 1)

            if return_boxes:
                assert self.needs_patches, "Can not return boxes if we do not need patches"
                return contains_handwriting, bboxes
            else:
                return contains_handwriting.any()

    def prepare_image(self, image:ImageClass) -> numpy.ndarray:
        image = image.convert('L').convert('RGB')
        if not self.needs_patches:
            image = image.resize(self.prediction_helper.image_size)
        image = numpy.array(image, dtype=chainer.get_dtype()).transpose(2, 0, 1)
        if self.needs_patches:
            image = self.prediction_helper.resize_image(image)
        image /= 255
        return image

    def normalize_box(self, box: AxisAlignedBBox, image_width: int, image_height: int) -> numpy.ndarray:
        normalized_box = AxisAlignedBBox(
            box.left / image_width,
            box.top / image_height,
            box.right / image_width,
            box.bottom / image_height
        )
        return normalized_box

    def get_analysis_grid(self, image: ImageClass, normalize_boxes: bool = False) -> List[dict]:
        assert self.needs_patches, "Can not return box grid if no patches are cropped"
        image = self.prepare_image(image)

        has_handwriting, boxes = self.evaluate_image(image, return_boxes=True)
        analysed = [
            {
                "has_handwriting": bool(handwriting_decision),
                "box": self.normalize_box(box, image.shape[-1], image.shape[-2]) if normalize_boxes else box
            }
            for handwriting_decision, box in zip(has_handwriting, boxes)
        ]
        return analysed

    def analyse_image(self, image: ImageClass) -> bool:
        image = self.prepare_image(image)

        has_handwriting = self.evaluate_image(image)
        return bool(chainer.backends.cuda.to_cpu(has_handwriting))

    def analyse_path(self, image_path: Path, file_name: str) -> dict:
        with Image.open(str(image_path)) as image:
            has_handwriting = self.analyse_image(image)

        return {
            "file_name": file_name,
            "has_handwriting": has_handwriting
        }
