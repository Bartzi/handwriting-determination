import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import celery
from PIL import Image
from celery import Celery

from analysis.analyzer import Analyzer


class AnalysisTask(celery.Task):

    def __init__(self):
        sys.path.append(str(Path(__file__).resolve().parent))
        self.config = self.load_config()
        self.config['model_path'] = os.environ.get('DETERMINATION_MODEL_PATH', self.config['model_path'])
        print(self.config)
        self.analyzer = None

    def initialize(self):
        if self.analyzer is not None:
            return
        self.analyzer = Analyzer(Path(self.config['model_path']), self.config['device'], needs_patches=True)

    def load_config(self) -> dict:
        with open('service_config.json') as f:
            config = json.load(f)
        return config

    # def run(self, task_data):
    #     image = base64.b85decode(task_data)
    #     io = BytesIO(image)
    #     io.seek(0)
    #
    #     with Image.open(io) as the_image:
    #         analyzed_boxes = self.analyzer.get_analysis_grid(the_image)
    #
    #     return json.dumps(analyzed_boxes)

# parser = argparse.ArgumentParser(description="run a service that takes images as inputs and returns analysis results")
# parser.add_argument('model', help="model to load")
# parser.add_argument("--max-size", type=int, default=2000, help="max size of input before splitting into patches")
# parser.add_argument("-d", "--device", default='@numpy', help="device to run on")
# parser.add_argument("-a", "--amqp-host", default='localhost', help="address of the messagequeue host to connect to")
#
# args = parser.parse_args()
#
# model_path = Path(args.model)
# device_id = -1 if args.device == '@numpy' else 1
# analyzer = Analyzer(model_path, device_id, needs_patches=True)


broker_address = os.environ.get('BROKER_ADDRESS', 'localhost')
app = Celery('determination', backend='rpc://', broker=f"pyamqp://guest@{broker_address}//")


@app.task(name='handwriting_determination', base=AnalysisTask)
def analyze(task_data):
    analyze.initialize()
    image = base64.b85decode(task_data['image'])
    io = BytesIO(image)
    io.seek(0)

    with Image.open(io) as the_image:
        analyzed_boxes = analyze.analyzer.get_analysis_grid(the_image)

    return json.dumps(analyzed_boxes)
# @app.task(name='handwriting_determination')
# def run_analysis(task_data):

