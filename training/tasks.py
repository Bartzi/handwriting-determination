import argparse
import msgpack
import os
import sys
from io import BytesIO
from pathlib import Path

import celery
from PIL import Image
from celery import Celery

from analysis.analyzer import Analyzer
import logging
logger = logging.getLogger(__file__)

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
app = Celery('wpi_demo', backend='rpc://', broker=f"pyamqp://guest@{broker_address}//")
app.conf.update(
    accept_content  = ['msgpack'],
    task_serializer = 'msgpack',
    result_serializer = 'msgpack',
)

@app.task(name='handwriting_determination', base=AnalysisTask, serializer="msgpack")
def analyze(task_data):
    analyze.initialize()
    bytes = msgpack.unpackb(task_data)

    image_data = BytesIO(bytes)
    image_data.seek(0)

    with Image.open(image_data) as the_image:
        the_image = the_image.convert('RGB')
        analyzed_boxes = analyze.analyzer.get_analysis_grid(the_image, normalize_boxes=True)
    return analyzed_boxes
