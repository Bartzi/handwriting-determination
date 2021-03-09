import argparse
import json
import multiprocessing
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import chainer
from PIL import Image
from tqdm import tqdm

from analysis.analyzer import is_image, Analyzer


def main(args):
    model_path = Path(args.model)
    root_dir = Path(args.root_dir)

    image_paths = [file_name for file_name in root_dir.glob('**/*') if is_image(file_name)]

    analyzed_images = []
    num_available_devices = chainer.backends.cuda.cupy.cuda.runtime.getDeviceCount()
    max_num_workers = max(num_available_devices, 1)
    analyser = Analyzer(model_path, max_num_workers, needs_patches=not args.no_split)
    ctx = multiprocessing.get_context('forkserver')

    # with ProcessPoolExecutor(max_workers=max_num_workers, mp_context=ctx) as executor:
    with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        current_jobs = []
        for i, image_path in enumerate(image_paths):
            submitted_job = executor.submit(analyser.analyse_path, image_path, i % max_num_workers, str(image_path.relative_to(root_dir)))
            current_jobs.append(submitted_job)

        for job in tqdm(as_completed(current_jobs), total=len(current_jobs)):
            try:
                result = job.result()
                analyzed_images.append(result)
            except Exception as e:
                print(f"Could not process {str(image_path)}, reason: {e}")

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

    main(parser.parse_args())
