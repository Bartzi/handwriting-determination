import argparse
import json
import multiprocessing
import sys
import traceback
from concurrent.futures._base import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import chainer
from tqdm import tqdm

from analysis.analyzer import is_image, Analyzer


def init_process(model_path, needs_patches, device):
    global analyzer
    current_process = multiprocessing.current_process()
    process_name = current_process.name
    if device == "@numpy":
        device_id = -1
    else:
        device_id = int(process_name.split('-')[-1]) - 1
    analyzer = Analyzer(model_path, device_id, needs_patches=needs_patches)


def consumer(image_path, file_name):
    return analyzer.analyse_path(image_path, file_name)


def main(args, device, num_available_devices):
    model_path = Path(args.model)
    root_dir = Path(args.root_dir)

    image_paths = [file_name for file_name in root_dir.glob('**/*') if is_image(file_name)]
    analyzed_images = []

    ctx = multiprocessing.get_context('forkserver')
    executor = ProcessPoolExecutor(max_workers=num_available_devices, mp_context=ctx, initializer=init_process, initargs=(model_path, not args.no_split, device))

    try:
        with executor:
            current_jobs = []
            for i, image_path in enumerate(image_paths):
                submitted_job = executor.submit(consumer, image_path, str(image_path.relative_to(root_dir)))
                current_jobs.append(submitted_job)

            for job in tqdm(as_completed(current_jobs), total=len(current_jobs)):
                try:
                    result = job.result()
                    analyzed_images.append(result)
                except Exception as e:
                    print(f"Could not process {str(image_path)}, reason: {e}")
                    traceback.print_exc(file=sys.stdout)
    except KeyboardInterrupt:
        pass

    with (root_dir / 'handwriting_analysis.json').open('w') as f:
        json.dump(analyzed_images, f, indent='\t')

    num_has_handwriting = len([im for im in analyzed_images if im['has_handwriting']])
    print(f"Handwriting to no handwriting ratio: {num_has_handwriting / len(analyzed_images)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provided a dir with images, create a json with info if an image contains handwriting or not")
    parser.add_argument("root_dir", help="path to dir to analyse")
    parser.add_argument('model', help="model to load")
    parser.add_argument("--max-size", type=int, default=2000, help="max size of input before splitting into patches")
    parser.add_argument("--no-split", action='store_true', default=False, help="do not split input image into individual patches")

    num_available_devices = chainer.backends.cuda.cupy.cuda.runtime.getDeviceCount()
    if num_available_devices == 0:
        num_available_devices = 1
        device = "@numpy"
    else:
        device = "@cuda"

    main(parser.parse_args(), device, num_available_devices)
