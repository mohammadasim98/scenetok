from pathlib import Path

import torch
import wandb


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_checkpoint(
    run_id: str,
    download_dir: Path,
    version: str | None,
) -> Path:
    api = wandb.Api()
    run = api.run(run_id)

    # Find the latest saved model checkpoint.
    chosen = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        # If no version is specified, use the latest.
        if version is None:
            if chosen is None or version_to_int(artifact) > version_to_int(chosen):
                chosen = artifact

        # If a specific verison is specified, look for it.
        elif version == artifact.version:
            chosen = artifact
            break

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
    chosen.download(root=root)
    return root / "model.ckpt"


def update_checkpoint_path(path: str | None, wandb_cfg: dict) -> Path | None:
    if path is None:
        return None

    if not str(path).startswith("wandb://"):
        return Path(path)

    run_id, *version = path[len("wandb://") :].split(":")
    if len(version) == 0:
        version = None
    elif len(version) == 1:
        version = version[0]
    else:
        raise ValueError("Invalid version specifier!")

    project = wandb_cfg["project"]
    return download_checkpoint(
        f"{project}/{run_id}",
        Path("checkpoints"),
        version,
    )


def log_tensor_as_video(logger, tensor, name, fps, step, caption):
    """
    Convert a (b, v, c, h, w) float tensor to wandb videos and log them.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (b, v, c, h, w) with values in [0, 1].
        name (str): Base name for the videos.
        fps (int): Frames per second for the video.
    """
    b, v, c, h, w = tensor.shape

    # Convert to numpy and rearrange to (b, v, h, w, c)
    # video = tensor.permute(0, 1, 3, 4, 2)  # (b, v, h, w, c)
    video = (tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    video_list = [video[i] for i in range(b)]
    fps_list = [fps] * b
    caption_list = caption
    # Log each video
    # for i in range(b):
    logger.log_video(
        key=f"{name}",
        videos=video_list,  # keep batch dimension
        fps=fps_list,
        caption=caption_list,
        step=step,
        format=["mp4"] * b
    )