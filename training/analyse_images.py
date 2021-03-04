import argparse
import json
import multiprocessing
from concurrent.futures._base import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import chainer
import cv2
import numpy
from PIL import Image
from chainer import Chain
from chainer.dataset import concat_examples
from cupy import get_array_module
from tqdm import tqdm

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

    def evaluate_image(self, image: numpy.ndarray, device_id: int) -> bool:
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
            contains_handwriting = (predicted_patches == 1).any()
        return contains_handwriting

    def analyse(self, image_path: Path, idx: int, file_name: str) -> dict:
        with Image.open(str(image_path)) as image:
            image = image.convert('L').convert('RGB')
            if not self.needs_patches:
                image = image.resize(self.prediction_helper.image_size)
            image = numpy.array(image, dtype=chainer.get_dtype()).transpose(2, 0, 1)
            if self.needs_patches:
                image = self.prediction_helper.resize_image(image)
            image /= 255

            has_handwriting = self.evaluate_image(image, idx)

        return {
            "file_name": file_name,
            "has_handwriting": bool(chainer.backends.cuda.to_cpu(has_handwriting))
        }


def main(args):
    model_path = Path(args.model)
    root_dir = Path(args.root_dir)

    image_paths = [file_name for file_name in root_dir.glob('**/*') if is_image(file_name)]

    analyzed_images = []
    num_available_devices = chainer.backends.cuda.cupy.cuda.runtime.getDeviceCount()
    max_num_workers = max(num_available_devices, 1)
    analyser = Analyzer(model_path, max_num_workers, needs_patches=not args.no_split)
    ctx = multiprocessing.get_context('forkserver')

    # with ProcessPoolExecutor(max_workers=max_num_workers, mp_context=ctx) as executor:
    with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        current_jobs = []
        for i, image_path in enumerate(image_paths):
            submitted_job = executor.submit(analyser.analyse, image_path, i % max_num_workers, str(image_path.relative_to(root_dir)))
            current_jobs.append(submitted_job)

        for job in tqdm(as_completed(current_jobs), total=len(current_jobs)):
            try:
                result = job.result()
                analyzed_images.append(result)
            except Exception as e:
                print(f"Could not process {str(image_path)}, reason: {e}")

    with (root_dir / 'handwriting_analysis.json').open('w') as f:
        json.dump(analyzed_images, f, indent='\t')

    num_has_handwriting = len([im for im in analyzed_images if im['has_handwriting']])
    print(f"Handwriting to no handwriting ratio: {num_has_handwriting / len(analyzed_images)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provided a dir with images, create a json with info if an image contains handwriting or not")
    parser.add_argument("root_dir", help="path to dir to analyse")
    parser.add_argument('model', help="model to load")
    parser.add_argument("--max-size", type=int, default=2000, help="max size of input before splitting into patches")
    parser.add_argument("--no-split", action='store_true', default=False, help="do not split input image into individual patches")

    main(parser.parse_args())
