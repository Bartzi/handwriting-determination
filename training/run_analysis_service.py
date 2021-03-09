import argparse
import base64
from io import BytesIO
from pprint import pprint

from PIL import Image
from celery import Celery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the analysis service by sending an image to analyse")
    parser.add_argument("image_path", help="Path of the image to analyse")
    parser.add_argument("-a", "--amqp-address", help="broker address")

    args = parser.parse_args()

    buffer = BytesIO()
    with Image.open(args.image_path) as image:
        image.save(buffer, format='PNG')

    data = {
        "image": base64.b85encode(buffer.getvalue(), pad=True).decode('utf-8'),
    }

    app = Celery('determination', backend='rpc://', broker=f"pyamqp://guest@{args.amqp_address}//")

    result = app.send_task('handwriting_determination', args=[data])
    pprint(result.get())
