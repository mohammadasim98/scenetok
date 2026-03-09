import os
import json
import torch
import numpy as np

from torch import Tensor
from pathlib import Path
from typing import TypedDict
from typing import Literal, List
from jaxtyping import Float, Int, UInt8


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]

def load_metadata(example_path: Path, scale_focal_by_256: bool=False) -> Metadata:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    url = str(example_path).split("/")[-3]
    with open(example_path, "r") as f:
        meta_data = json.load(f)

    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    saved_fx = float(fx) / float(store_w)
    saved_fy = float(fy) / float(store_h)
    saved_cx = float(cx) / float(store_w)
    saved_cy = float(cy) / float(store_h)

    # NOTE: Know bug for the trained va-videodc model
    # Normally we wont scale it since intrinsics are supposed to be normalized
    if scale_focal_by_256:
        saved_fx *= float(store_w) / 256
        saved_fy *= float(store_h) / 256

    timestamps = []
    cameras = []
    opencv_c2ws = []  # will be used to calculate camera distance

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
        opencv_c2ws.append(opencv_c2w)
        camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
        cameras.append(np.array(camera))

    # timestamp should be the one that match the above images keys, use for indexing
    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {"url": url, "timestamps": timestamps, "cameras": cameras}