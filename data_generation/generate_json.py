import argparse
import json
import os
import random

from tqdm import tqdm


def create_gt(image_paths, destination, has_text):
    image_data = []
    for image_path in tqdm(image_paths):
        image_data.append({
            "file_name": os.path.relpath(image_path, start=os.path.dirname(destination)),
            "has_handwriting": has_text
        })
    return image_data


def image_filter(x):
    return [os.path.join(args.data_dir, x, image) for image in os.listdir(os.path.join(args.data_dir, x)) if os.path.splitext(image)[-1] == ".png"]


def main(args):
    positive_images = image_filter('positive')
    negative_images = image_filter('negative')

    positive_gt = create_gt(positive_images, args.destination, True)
    negative_gt = create_gt(negative_images, args.destination, False)

    gt = positive_gt + negative_gt
    random.shuffle(gt)

    with open(args.destination, 'w') as f:
        json.dump(gt, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that generates a json file for the chainer implementation from generated data")
    parser.add_argument("data_dir", help="path to dir with the data json shall be generated from")
    parser.add_argument("--destination", help="overwrite default destination of json file (default is same dir as data_dir)")


    args = parser.parse_args()
    if args.destination is None:
        args.destination = os.path.join(os.path.realpath(args.data_dir), "gt.json")

    main(args)
