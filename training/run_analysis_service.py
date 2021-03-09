import argparse
import base64
import json
import sys
from io import BytesIO
from pprint import pprint

import pika
from PIL import Image


def build_return_channel(host, channel_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()

    channel.queue_declare(queue=channel_name, durable=True)
    return channel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the analysis service by sending an image to analyse")
    parser.add_argument("image_path", help="Path of the image to analyse")
    parser.add_argument("-a", "--amqp-host", default='localhost', help="address of message broker")

    args = parser.parse_args()

    buffer = BytesIO()
    with Image.open(args.image_path) as image:
        image.save(buffer, format='PNG')

    return_channel_name = "test123"
    data = {
        "image": base64.b85encode(buffer.getvalue(), pad=True).decode('utf-8'),
        "return_channel": return_channel_name,
    }

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=args.amqp_host)
    )
    channel = connection.channel()

    channel.queue_declare(queue='determination', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='determination',
        body=json.dumps(data),
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )
    print(" [x] Sent Message")
    connection.close()

    return_channel = build_return_channel(args.amqp_host, return_channel_name)

    def callback(ch, method, properties, body):
        data = json.loads(body.decode('utf-8'))
        pprint(data)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    return_channel.basic_consume(queue=return_channel_name, on_message_callback=callback)
    try:
        return_channel.start_consuming()
    except KeyboardInterrupt:
        sys.exit(0)
