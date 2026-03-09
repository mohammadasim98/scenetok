import io
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import imageio

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

from IPython.display import HTML
import base64


def plot_tensor_row(video_tensor, figsize=(15, 3), savefig: bool=False, name: str="scene"):
    """
    Plot a tensor of shape (N, C, H, W) as a row of images.
    
    Args:
        video_tensor: torch.Tensor or np.ndarray, shape (N, C, H, W), values in [0,1].
        figsize: size of the matplotlib figure.
    """
    # Convert torch → numpy
    if "torch" in str(type(video_tensor)):
        video_tensor = video_tensor.detach().cpu().numpy()
    
    # Clamp and scale to [0,1]
    video_tensor = np.clip(video_tensor, 0, 1)

    # Handle grayscale (C=1) and RGB (C=3)
    if video_tensor.shape[1] == 1:
        video_tensor = video_tensor[:, 0]  # (N, H, W)
    elif video_tensor.shape[1] == 3:
        video_tensor = np.transpose(video_tensor, (0, 2, 3, 1))  # (N, H, W, 3)
    else:
        raise ValueError("Only supports C=1 or C=3 tensors.")

    N = video_tensor.shape[0]
    fig, axes = plt.subplots(1, N, figsize=figsize)
    if N == 1:
        axes = [axes]  # make iterable
    
    for i, ax in enumerate(axes):
        ax.imshow(video_tensor[i], cmap="gray" if video_tensor.ndim == 3 else None)
        ax.axis("off")
    plt.show()
    if savefig:
        plt.imsave(f"depth_{name}.png", np.concatenate([video_tensor[i]*255 for i in range(len(axes))], axis=1).astype(np.uint8))


def plot_tensor_col(video_tensor, figsize=(15, 3)):
    """
    Plot a tensor of shape (N, C, H, W) as a row of images.
    
    Args:
        video_tensor: torch.Tensor or np.ndarray, shape (N, C, H, W), values in [0,1].
        figsize: size of the matplotlib figure.
    """
    # Convert torch → numpy
    if "torch" in str(type(video_tensor)):
        video_tensor = video_tensor.detach().cpu().numpy()
    
    # Clamp and scale to [0,1]
    video_tensor = np.clip(video_tensor, 0, 1)

    # Handle grayscale (C=1) and RGB (C=3)
    if video_tensor.shape[1] == 1:
        video_tensor = video_tensor[:, 0]  # (N, H, W)
    elif video_tensor.shape[1] == 3:
        video_tensor = np.transpose(video_tensor, (0, 2, 3, 1))  # (N, H, W, 3)
    else:
        raise ValueError("Only supports C=1 or C=3 tensors.")

    N = video_tensor.shape[0]
    fig, axes = plt.subplots(N, 1, figsize=figsize)
    if N == 1:
        axes = [axes]  # make iterable
    
    for i, ax in enumerate(axes):
        ax.imshow(video_tensor[i], cmap="gray" if video_tensor.ndim == 3 else None)
        ax.axis("off")

    plt.show()

def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img

def play_tensor_depth_video(video_tensor, fps=24):
    """
    Plays a video in Jupyter from a numpy/tensor array of shape (N, C, H, W).
    
    Args:
        video_tensor: numpy array or torch tensor of shape (N, C, H, W), values in [0, 1].
        fps: frames per second for playback.
    """
    # If torch tensor, move to cpu + convert to numpy
    if "torch" in str(type(video_tensor)):
        video_tensor = video_tensor.detach().cpu().numpy()
    *_, h, w = video_tensor.shape
    # Ensure values are in [0,255] and type uint8
    video_tensor = np.clip(video_tensor, 0, 1)
    video_tensor = (video_tensor * 255).astype(np.uint8)

    # Convert to (N, H, W, C)
    # if video_tensor.shape[1] == 1:  # grayscale
    #     video_tensor = np.repeat(video_tensor, 3, axis=1)
    video_tensor = np.transpose(video_tensor, (0, 2, 3, 1))
    video_tensor_list = []
    for tensor in video_tensor:
        video_tensor_list.append(colorize(tensor[..., 0]))
    video_tensor = np.stack(video_tensor_list, axis=0)
    # Write video to memory
    with io.BytesIO() as f:
        imageio.mimsave(f, video_tensor, format="mp4", fps=fps)
        video_bytes = f.getvalue()

    # Encode to base64 for HTML
    video_b64 = base64.b64encode(video_bytes).decode()
    html = f"""
    <video width={w} height={h} controls autoplay loop muted>
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    return HTML(html)

def play_tensor_video(video_tensor, fps=24):
    """
    Plays a video in Jupyter from a numpy/tensor array of shape (N, C, H, W).
    
    Args:
        video_tensor: numpy array or torch tensor of shape (N, C, H, W), values in [0, 1].
        fps: frames per second for playback.
    """
    # If torch tensor, move to cpu + convert to numpy
    if "torch" in str(type(video_tensor)):
        video_tensor = video_tensor.detach().cpu().numpy()
    *_, h, w = video_tensor.shape
    # Ensure values are in [0,255] and type uint8
    video_tensor = np.clip(video_tensor, 0, 1)
    video_tensor = (video_tensor * 255).astype(np.uint8)

    # Convert to (N, H, W, C)
    if video_tensor.shape[1] == 1:  # grayscale
        video_tensor = np.repeat(video_tensor, 3, axis=1)
    video_tensor = np.transpose(video_tensor, (0, 2, 3, 1))

    # Write video to memory
    with io.BytesIO() as f:
        imageio.mimsave(f, video_tensor, format="mp4", fps=fps)
        video_bytes = f.getvalue()

    # Encode to base64 for HTML
    video_b64 = base64.b64encode(video_bytes).decode()
    html = f"""
    <video width={w} height={h} controls autoplay loop muted>
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    return HTML(html)
def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> Union[
    UInt8[np.ndarray, "height width 3"],
    UInt8[np.ndarray, "height width 4"]
]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]

def get_hist_image(x, title, bins=50, dpi=100, figsize=(5, 4)):
    
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    
    # Do some plotting here
    ax = fig.add_subplot(111)
    data =torch.clamp(x, min=-5, max=5).detach().cpu().flatten().numpy()
    ax.hist(data, bins=None)
    ax.axis('on')
    # ax.set_xlim(-7, 7)
    # ax.set_ylim(0, 500)
    ax.set_title(title)
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    
    rgba = np.asarray(buf)
    row, col, ch = rgba.shape
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    background=(255,255,255)
    R, G, B = background
    rgb[:,:,0] = r #* a + (1.0 - a) * R
    rgb[:,:,1] = g #* a + (1.0 - a) * G
    rgb[:,:,2] = b #* a + (1.0 - a) * B
    rgb = np.asarray( rgb, dtype='uint8')
    # convert to a NumPy array
    return torch.tensor(rgb/(255.0)).permute(2, 0, 1)




# def fixed_padding(inputs, kernel_size, dilation):
#     kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
#     pad_total = kernel_size_effective - 1
#     pad_beg = pad_total // 2
#     pad_end = pad_total - pad_beg
#     padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
#     return padded_inputs

def prep_image(image: FloatImage) -> Union[
    UInt8[np.ndarray, "height width 3"],
    UInt8[np.ndarray, "height width 4"]
]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def prep_video(video) -> UInt8[np.ndarray, "frame 3 height width"]:
    # Handle single-channel videos.
    if video.ndim == 3:
        video = rearrange(video, "f h w -> f () h w")

    # Ensure that there are 3 channels.
    _, channel, _, _ = video.shape
    if channel == 1:
        video = repeat(video, "f () h w -> f c h w", c=3)
    assert video.shape[1] == 3

    video = (video.detach().clip(min=0, max=1) * 255)\
        .to(dtype=torch.uint8, device="cpu")
    return video.numpy()

# def save_video(
#     video: UInt8[np.ndarray, "frame 3 height width"],
#     path: Union[Path, str],
#     **kwargs
# ) -> None:
#     """Save an image. Assumed to be in range 0-1."""
#     # Create the parent directory if it doesn't already exist.
#     path = Path(path)
#     path.parent.mkdir(exist_ok=True, parents=True)
#     # Save the video.
#     write_video(str(path), video.transpose(0, 2, 3, 1), **kwargs)
    
    
def save_image_video(images, indices, output_dir, name, save_img: bool=True, save_video: bool=True):
    
    os.makedirs(output_dir, exist_ok=True)
    if save_img:
        for k, (color, idx) in enumerate(zip(images, indices)):
            save_image(color, Path(output_dir) / f"{int(idx):0>6}.png") 

    # directory = output_dir / scene / name
    
    # original_image_pil = [Image.open(f"{directory}/{fi}").convert("RGB") for fi in sorted(os.listdir(directory))]
    # original_image_pil[0].save(
    #     output_dir / scene / f"{name}.gif",
    #     save_all=True,
    #     append_images=original_image_pil[1:],
    #     duration=5,
    #     loop=0 # 0 means infinite loop
    #     )
    if save_video:
        images = images.permute(0, 2, 3, 1) * 255
        original_image_pil = [np.asarray(img) for img in images.cpu()]
        h_fps = ImageSequenceClip(original_image_pil, fps=24)
        # l_fps = ImageSequenceClip(original_image_pil, fps=10)
        h_fps.write_videofile(str(output_dir / f"{name}.mp4"),fps=24)
    
    # l_fps.write_videofile(str(output_dir / scene / f"{name}_10.mp4"),fps=10)