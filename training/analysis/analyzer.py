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

Image.init()


def is_image(file_name: Path) -> bool:
    if not isinstance(file_name, Path):
        file_name = Path(file_name)
    return file_name.suffix.lower() in Image.EXTENSION.keys()


class Analyzer:

    def __init__(self, model_path: Path, max_num_workers: int, needs_patches: bool = True):
        self.model_path = model_path
        self.max_num_workers = max_num_workers
        self.needs_patches = needs_patches

        log_file = self.model_path.parent / 'log'

        with log_file.open() as f:
            self.log_data = json.load(f)[0]

        self.prediction_helper = PredictionDatasetMixin(image_size=self.log_data['image_size'], max_size=2000)
        self.initialized = False
        self.initialize()
        self.network = None

    def initialize(self):
        if self.initialized:
            return

        self.initialized = True

        if self.max_num_workers >= 0:
            self.networks = []
            for i in range(self.max_num_workers):
                with chainer.using_device(chainer.get_device(i)):
                    self.networks.append(self.load_network(i))
        else:
            self.networks = [self.load_network('@numpy')]

    def load_network(self, device_id) -> Chain:
        net_class = restore_backup(self.log_data['net'], '.')
        net = net_class()

        with numpy.load(str(self.model_path)) as f:
            chainer.serializers.NpzDeserializer(f, strict=True).load(net)

        net = net.to_device(device_id)

        return net

    def evaluate_image(self, image: numpy.ndarray, device_id: int, return_boxes: bool = False) -> Union[bool, Tuple[bool, list]]:
        if self.needs_patches:
            patches, bboxes = self.prediction_helper.create_sliding_window(image)
        else:
            patches = image[numpy.newaxis, ...]

        network = self.networks[device_id]
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
            contains_handwriting = predicted_patches == 1
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

    def get_analysis_grid(self, image: ImageClass, idx: int = 0) -> List[dict]:
        assert self.needs_patches, "Can not return box grid if no patches are cropped"
        image = self.prepare_image(image)

        has_handwriting, boxes = self.evaluate_image(image, idx, return_boxes=True)
        analysed = [
            {
                "has_handwriting": bool(handwriting_decision),
                "box": box
            }
            for handwriting_decision, box in zip(has_handwriting, boxes)
        ]
        return analysed

    def analyse_image(self, image: ImageClass, idx: int) -> bool:
        image = self.prepare_image(image)

        has_handwriting = self.evaluate_image(image, idx)
        return bool(chainer.backends.cuda.to_cpu(has_handwriting))

    def analyse_path(self, image_path: Path, idx: int, file_name: str) -> dict:
        with Image.open(str(image_path)) as image:
            has_handwriting = self.analyse_image(image, idx)

        return {
            "file_name": file_name,
            "has_handwriting": has_handwriting
        }
