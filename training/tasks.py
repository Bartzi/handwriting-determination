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
        self.config = {
            'model_path': os.environ.get('DETERMINATION_MODEL_PATH', None),
            'device_id': int(os.environ.get('DETERMINATION_DEVICE', -1))
        }
        print(self.config)
        assert self.config['model_path'] is not None, "You must supply a model in the environment variable DETERMINATION_MODEL_PATH"
        self.analyzer = None

    def initialize(self):
        if self.analyzer is not None:
            return
        self.analyzer = Analyzer(Path(self.config['model_path']), self.config['device_id'], needs_patches=True)


broker_address = os.environ.get('BROKER_ADDRESS', 'localhost')
app = Celery('determination', backend='rpc://', broker=f"pyamqp://guest@{broker_address}//")


@app.task(name='handwriting_determination', base=AnalysisTask)
def analyze(task_data):
    analyze.initialize()
    image = base64.b85decode(task_data['image'])
    io = BytesIO(image)
    io.seek(0)

    with Image.open(io) as the_image:
        analyzed_boxes = analyze.analyzer.get_analysis_grid(the_image, normalize_boxes=True)

    return analyzed_boxes
