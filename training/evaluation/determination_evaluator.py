import os

import chainer
import chainer.functions as F
import numpy
from PIL import ImageDraw
from chainer.backend import get_array_module

from datasets.prediction_dataset import PredictionDataset
from evaluation.evaluator import Evaluator
from utils.bbox.bbox import AxisAlignedBBox


class DeterminationEvaluator(Evaluator):

    def build_dataset(self, args):
        return PredictionDataset(
            args.eval_gt,
            args.image_size,
            root=os.path.dirname(args.eval_gt),
            dtype=chainer.get_dtype(),
            image_mode=args.image_mode,
            max_size=2000,
        )

    def preprocess_batch(self, batch):
        xp = get_array_module(batch['image'])
        batch_size, num_windows, num_channels, window_height, window_width = batch['image'].shape
        batch['image'] = xp.reshape(batch['image'], (-1, num_channels, window_height, window_width))

        return batch, num_windows

    def run_network(self, batch, num_windows):
        xp = get_array_module(batch['image'])
        predictions = self.net(**batch)
        predictions = xp.reshape(predictions.array, (-1, num_windows) + predictions.shape[1:])

        return predictions

    def handwriting_or_not(self, predictions):
        xp = get_array_module(predictions)

        contains_handwriting = []
        for image in predictions:
            predicted_classes = xp.argmax(image, axis=1)
            contains_handwriting.append((predicted_classes == 1).any())

        return xp.stack(contains_handwriting, axis=0)

    def render_regions(self, batch, image_has_handwriting, window_has_handwriting, batch_id):
        xp = get_array_module(window_has_handwriting)
        images = batch['original_image']
        boxes = batch['boxes']
        labels = batch['has_text']

        window_has_handwriting = F.softmax(window_has_handwriting, axis=2).array

        iterator = zip(
            labels,
            images,
            image_has_handwriting,
            window_has_handwriting,
            boxes
        )
        for i, (label, image, has_handwriting, windows_with_handwriting, boxes) in enumerate(iterator):
            render_dir = self.determine_prediction_type(label, has_handwriting)
            if self.region_filter != 'all':
                if render_dir != self.region_filter:
                    continue

            image = self.array_to_image(image)
            draw = ImageDraw.Draw(image)

            for window, box in zip(windows_with_handwriting, boxes):
                predicted_class = int(xp.argmax(window))
                color = self.render_colors[predicted_class]
                cpu_box = numpy.empty(box.shape, dtype=box.dtype)
                chainer.backend.copyto(cpu_box, box)
                cpu_box = AxisAlignedBBox(*cpu_box.tolist())
                self.draw_box(cpu_box, draw, color)
                self.render_confidence(
                    cpu_box,
                    draw,
                    float(window[predicted_class]),
                    color,
                )

            image.save(os.path.join(self.render_dir, render_dir, f"{batch_id}_{i}.png"))
