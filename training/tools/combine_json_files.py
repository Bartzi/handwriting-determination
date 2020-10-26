import argparse
import json
import os

import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that combines multiple json gt files into one")
    parser.add_argument("json_files", nargs="+", help="json files to combine")

    args = parser.parse_args()

    json_data = []
    for json_file in args.json_files:
        with open(json_file) as f:
            data = json.load(f)
        json_data.extend(data)

    random.shuffle(json_data)

    with open(os.path.join(os.path.dirname(args.json_files[0]), "combined.json"), 'w') as f:
        json.dump(json_data, f, indent=2)
