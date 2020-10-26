import os

import cv2
import numpy as np
import random
import threading, queue
import argparse
from tqdm import tqdm

from fragment_generator import generate_fragment, positive_trait, negative_trait

DIMENSION_X, DIMENSION_Y = 224, 224


# write images asynchronously to disk to unblock computation
to_write = queue.Queue(maxsize=10000)


def writer():
    """
    Writer for the writer thread that saves generated fragments to disk
    """
    # Call to_write.get() until it returns None
    for write_task in iter(to_write.get, None):
        dirname = os.path.dirname(write_task[0])
        os.makedirs(dirname, exist_ok=True)

        success = cv2.imwrite(write_task[0], write_task[1])
        if not success:
            raise RuntimeError(f"Could not save generated sample to {write_task[0]}")


def write_fragments(trait_generator, amount, path):
    """
    Generates fragments with the traits given by the given trait generator and saves them to the given path
    """
    # existing_files = [os.path.join(root, name)
    #                     for root, dirs, files in os.walk(path)
    #                     for name in files if name.endswith(IMAGE_FORMATS)]
    existing_files = []

    for i in tqdm(range(len(existing_files), amount), total=amount - len(existing_files), ascii=True):
        traits = trait_generator()
        fragment = generate_fragment(traits, (DIMENSION_X, DIMENSION_Y))

        file_path = path + "/{}.png".format(i)
        write_task = (file_path, fragment)
        to_write.put(write_task)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--positive", default=100, type=int,
                    help="Amount of positive fragments that should be generated (default is 100).")
    ap.add_argument("-n", "--negative", default=100, type=int,
                    help="Amount of negative fragments that should be generated (default is 100)/")
    ap.add_argument("-pp", "--pospath", default="./generated/positive",
                    help="Path to which positive fragments will be saved.")
    ap.add_argument("-np", "--negpath", default="./generated/negative",
                    help="Path to which negative fragments will be saved.")
    args = vars(ap.parse_args())

    positive_path = args['pospath']
    negative_path = args['negpath']
    positive_amount = args['positive']
    negative_amount = args['negative']

    # start the writer for the fragments
    threading.Thread(target=writer).start()

    write_fragments(positive_trait, positive_amount, positive_path)
    write_fragments(negative_trait, negative_amount, negative_path)
    print("Generation is done, waiting for writing...")

    # enqueue None to instruct the writer thread to exit
    to_write.put(None)
