


import os

import json
from pathlib import Path
import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root", type=str, help="Root directory of the raw dataset")
    args = argparser.parse_args()
    root = Path(args.root)
    folders = os.listdir(root / "train")

    _dict = {}
    for stage in ["train", "test"]:

        _dict[stage] = {}

        for downsample in os.listdir(root / stage):
            _dict[stage][downsample] = {}
            for file in os.listdir(root / stage / downsample):
                if "npz" in file and "flipped" not in file:
                    name = file.split(".")[0]
                    ext = file.split(".")[1]
                    _dict[stage][downsample][name] = [file, f"{name}_flipped.{ext}"]


    with open(root / "map_dict.json", "w") as f:

        f.write(json.dumps(_dict))

