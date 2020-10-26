import argparse

import json

import copy
import os

import random


def main(args):
    with open(args.gt_file) as f:
        gt_data = json.load(f)

    images = gt_data.pop('images')
    random.shuffle(images)

    num_validation_images = int(len(images) * args.val_ratio)
    validation_images = images[:num_validation_images]
    train_images = images[num_validation_images:]

    train_data = copy.copy(gt_data)
    train_data['images'] = train_images

    validation_data = copy.copy(gt_data)
    validation_data['images'] = validation_images

    base_dir = os.path.dirname(args.gt_file)

    with open(os.path.join(base_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(base_dir, 'val.json'), 'w') as f:
        json.dump(validation_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool that takes gt and creates train and val split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("gt_file", help="path to gt file that is to be split")
    parser.add_argument("-v", "--val-ratio", type=float, default=0.2, help="ratio of validation files")

    args = parser.parse_args()

    main(args)
