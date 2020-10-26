import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that converts WPI GT to GT used in this project")
    parser.add_argument("gt_files", nargs='+')

    args = parser.parse_args()

    for gt_file in args.gt_files:
        with open(gt_file) as f:
            gt_data = json.load(f)

        new_gt = []
        for image_data in gt_data:
            handwriting_key = 'handwriting' if 'handwriting' in image_data else 'has_handwriting'
            new_gt.append({
                "file_name": image_data['file_name'],
                "has_handwriting": image_data[handwriting_key],
            })

        with open(gt_file, 'w') as f:
            json.dump(new_gt, f, indent=2)
