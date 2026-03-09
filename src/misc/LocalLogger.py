import os
from pathlib import Path
from typing import Any, Optional
import torch
from PIL import Image
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from torchvision.io import write_video

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.experiment = None
        os.system(f"rm -r {LOG_PATH}")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = LOG_PATH / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)

    
    @rank_zero_only
    def log_video(
        self,
        key: str,
        videos: list[Any],
        fps: list[int],
        caption: list[str],
        step: Optional[int],
        format: list[str],
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        
        assert step is not None
        for index, video in enumerate(videos):
            fmat= format[index]
            cap = caption[index]
            path = LOG_PATH / f"{key}/{cap}_{step:0>6}.{fmat}"
            path.parent.mkdir(exist_ok=True, parents=True)
            video = torch.from_numpy(video).permute(0, 2, 3, 1).cpu()
            print(video.shape, video.dtype)
            print(path)
            print(cap)
            print(fmat)
            print(fps[index])
            write_video(
                filename=path,
                video_array=video,
                fps=fps[index],
                video_codec='libx264',
                options={"crf": "20"}
            )
            
