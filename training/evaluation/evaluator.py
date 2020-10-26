import json
from collections import defaultdict
from pprint import pprint

import chainer
import matplotlib.pyplot as plt
import numpy
import os
import random
import re
from PIL import ImageFont, Image
from chainer import reporter as reporter_module
from chainer.backend import get_array_module
from chainer.dataset import concat_examples
from tqdm import tqdm

from config.config import parse_config
from datasets.prediction_dataset import PredictionDataset
from utils.backup import restore_backup
from utils.bbox.bbox import AxisAlignedBBox


class Evaluator:

    def __init__(self, args):
        with open(os.path.join(args.log_dir, args.log_name)) as f:
            log_metadata = json.load(f)[0]

        self.args = self.find_and_parse_config(args, log_metadata)
        chainer.global_config.dtype = eval(self.args.dtype)

        self.data_loader = self.build_dataset(args)

        self.data_iter = chainer.iterators.MultithreadIterator(
            self.data_loader,
            batch_size=args.eval_batch_size,
            shuffle=False,
            repeat=False,
            # n_prefetch=2
        )

        net_class = restore_backup(log_metadata['net'], args.log_dir)
        self.net = net_class()
        self.net.to_device(args.gpu)

        self.results_path = os.path.join(args.log_dir, f"{args.evaluation_name}.json")
        self.current_snapshot = None

        if self.args.render_regions:
            self.font = ImageFont.truetype("utils/DejaVuSans.ttf", 20)
            self.region_filter = self.args.render_regions
            self.base_render_dir = os.path.join(self.args.log_dir, 'eval_renderings')
            self.render_colors = ['red', 'green']
            self.render_index = 0 if self.args.render_negatives else 1
            os.makedirs(self.base_render_dir, exist_ok=True)

    def build_dataset(self, args):
        raise NotImplementedError

    def filter_snapshots(self, prefix):
        evaluated_snapshots = []
        if os.path.exists(self.results_path):
            if self.args.force_reset:
                os.unlink(self.results_path)
            else:
                with open(self.results_path) as f:
                    json_data = json.load(f)
                    evaluated_snapshots = [item['snapshot_name'] for item in json_data]

        snapshots = list(
            sorted(
                filter(lambda x: x not in evaluated_snapshots and prefix in x,
                       os.listdir(self.args.log_dir)),
                key=lambda x: int(getattr(re.search(r"(\d+).npz", x), 'group', lambda: 0)(1))
            )
        )
        return snapshots

    def find_and_parse_config(self, args, log_contents):
        config_file = os.path.join(args.log_dir, 'code', log_contents['config'])
        return parse_config(config_file, args)

    def determine_prediction_type(self, label, prediction):
        if label:
            if prediction:
                return "tp"
            else:
                return "fn"
        else:
            if prediction:
                return "fp"
            else:
                return "tn"

    def calc_metrics(self, predictions, labels):
        xp = get_array_module(predictions, labels)
        labels = labels.astype(xp.bool)

        metrics = defaultdict(int)
        for image_prediction, label in zip(predictions, labels):
            metrics[self.determine_prediction_type(label, image_prediction)] += 1

        return metrics

    def calc_precision_and_recall(self, summary):
        true_positives = summary._summaries['tp']._x
        true_negatives = summary._summaries['tn']._x
        false_positives = summary._summaries['fp']._x
        false_negatives = summary._summaries['fn']._x

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        false_positive_rate = false_positives / max(false_positives + true_negatives, 1)
        false_negative_rate = false_negatives / max(false_negatives + true_positives, 1)

        f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }

    def plot_eval_results(self, data):
        def get_colors(n):
            ret = []
            r = int(random.random() * 256)
            g = int(random.random() * 256)
            b = int(random.random() * 256)
            step = 256 / n
            for i in range(n):
                r += step
                g += step
                b += step
                r = int(r) % 256
                g = int(g) % 256
                b = int(b) % 256
                ret.append((r, g, b))
            return ret

        values_per_key = defaultdict(list)

        for element in data:
            for key, value in element.items():
                values_per_key[key] += [value]

        colors = get_colors(len(list(values_per_key.keys())))
        for (key, value), color in zip(values_per_key.items(), colors):
            if key == 'snapshot_name':
                continue
            plt.plot(value, label=key)

        plt.legend()
        plt.savefig(os.path.join(self.args.log_dir, f"plot_{self.args.evaluation_name}.png"))

    def find_best_result(self, json_data):
        best_snapshot = max(json_data, key=lambda x: x['f1_score'])
        with open(os.path.join(self.args.log_dir, "best_evaluation_result.json"), 'w') as f:
            json.dump(best_snapshot, f, indent=2)
        pprint(best_snapshot)

    def save_eval_result(self, eval_result, plot_and_print_only=False):
        if os.path.exists(self.results_path):
            with open(self.results_path) as f:
                json_data = json.load(f)
        else:
            json_data = []

        if plot_and_print_only:
            self.plot_eval_results(json_data)
            self.find_best_result(json_data)
        else:
            json_data.append(eval_result)
            with open(self.results_path, 'w') as f:
                json.dump(json_data, f, indent=2)

    def load_weights(self, snapshot_name):
        self.current_snapshot = snapshot_name

        if self.args.render_regions:
            self.render_dir = os.path.join(self.base_render_dir, snapshot_name)
            for name in ['tp', 'tn', 'fp', 'fn']:
                os.makedirs(os.path.join(self.render_dir, name), exist_ok=True)

        with numpy.load(os.path.join(self.args.log_dir, snapshot_name)) as f:
            chainer.serializers.NpzDeserializer(f, strict=True).load(self.net)
        self.data_iter.reset()

    def preprocess_batch(self, batch):
        raise NotImplementedError

    def run_network(self, batch, num_windows):
        raise NotImplementedError

    def handwriting_or_not(self, predictions):
        raise NotImplementedError

    def render_regions(self, batch, image_has_handwriting, window_has_handwriting, batch_id):
        raise NotImplementedError

    def evaluate(self):
        summary = reporter_module.DictSummary()
        current_device = chainer.get_device(self.args.gpu)

        with chainer.using_device(current_device), chainer.configuration.using_config('train', False):
            for i, batch in enumerate(tqdm(self.data_iter, total=len(self.data_loader) // self.args.eval_batch_size, leave=False)):
                batch = concat_examples(batch, self.args.gpu)

                batch, num_windows = self.preprocess_batch(batch)
                predictions = self.run_network(batch, num_windows)
                handwriting_predictions = self.handwriting_or_not(predictions)

                if self.args.render_regions:
                    self.render_regions(batch, handwriting_predictions, predictions, i)

                summary.add(self.calc_metrics(handwriting_predictions, batch["has_text"]))

        eval_result = self.calc_precision_and_recall(summary)
        eval_result['snapshot_name'] = self.current_snapshot

        self.save_eval_result(eval_result)

    def draw_box(self, box, draw, color):
        corners = [(box.left, box.top), (box.right, box.top), (box.right, box.bottom), (box.left, box.bottom)]
        next_corners = corners[1:] + [corners[0]]

        for first_corner, next_corner in zip(corners, next_corners):
            draw.line([first_corner, next_corner], fill=color, width=3, joint='curve')

    def render_confidence(self, box, draw, confidence, color):
        text = format(confidence, ".3f")
        text_width, text_height = draw.textsize(text, font=self.font)
        text_box = AxisAlignedBBox(
            left=box.right - box.width // 2 - text_width // 2,
            top=box.top,
            right=box.right - box.width // 2 + text_width // 2,
            bottom=box.top + text_height,
        )
        draw.rectangle(text_box, fill=(255, 255, 255, 160))
        draw.text((text_box.left, text_box.top), text, fill=color, font=self.font)

    def array_to_image(self, image):
        cpu_array = numpy.empty(image.shape, dtype=image.dtype)
        chainer.backend.copyto(cpu_array, image)
        if len(image.shape) == 3:
            image = (cpu_array.transpose(1, 2, 0) * 255).astype(numpy.uint8)
        else:
            image = (cpu_array * 255).astype(numpy.uint8)
        return Image.fromarray(image)
