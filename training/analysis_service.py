import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import pika
from PIL import Image

from analysis.analyzer import Analyzer


class AnalysisService:

    def __init__(self, model_path: Path, device_id: int, amqp_host: str):
        self.analyser = Analyzer(model_path, device_id, needs_patches=True)
        self.amqp_host = amqp_host

    def open_channel(self, channel_name: str):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.amqp_host)
        )
        channel = connection.channel()

        channel.queue_declare(queue=channel_name, durable=True)
        return connection, channel

    def run_analysis(self, ch, method, properties, body):
        data = json.loads(body.decode('utf-8'))
        image = base64.b85decode(data['image'])
        io = BytesIO(image)
        io.seek(0)

        with Image.open(io) as the_image:
            analyzed_boxes = self.analyser.get_analysis_grid(the_image, 0)

        connection, return_channel = self.open_channel(data['return_channel'])
        return_channel.basic_publish(
            exchange='',
            routing_key=data['return_channel'],
            body=json.dumps(analyzed_boxes),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )
        connection.close()
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main(args: argparse.Namespace):
    model_path = Path(args.model)
    device_id = -1 if args.device == '@numpy' else 1
    service = AnalysisService(model_path, device_id, args.amqp_host)

    connection = pika.BlockingConnection(pika.ConnectionParameters(host=args.amqp_host))
    channel = connection.channel()

    channel.queue_declare(queue='determination', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='determination', on_message_callback=service.run_analysis)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run a service that takes images as inputs and returns analysis results")
    parser.add_argument('model', help="model to load")
    parser.add_argument("--max-size", type=int, default=2000, help="max size of input before splitting into patches")
    parser.add_argument("-d", "--device", default='@numpy', help="device to run on")
    parser.add_argument("-a", "--amqp-host", default='localhost', help="address of the messagequeue host to connect to")

    main(parser.parse_args())
