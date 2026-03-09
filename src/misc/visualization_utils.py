import torch
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def reshape_tokens_to_image(x: torch.Tensor, H: int = None, W: int = None) -> torch.Tensor:
    """
    Reshape a tensor of shape (B, N, C) to (B, H, W, C), where H * W = N.
    - N must be a power of 2.
    - If H and W not provided, selects the most balanced (H, W) pair such that both are powers of 2.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C)
        H (int, optional): Desired height
        W (int, optional): Desired width

    Returns:
        torch.Tensor: Reshaped tensor of shape (B, H, W, C)
    """
    B, N, C = x.shape

    # Check N is power of 2
    if (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2")

    if H is not None and W is not None:
        if H * W != N:
            raise ValueError(f"H * W = {H * W} != N = {N}")
        if (H & (H - 1)) != 0 or (W & (W - 1)) != 0:
            raise ValueError("H and W must be powers of 2")
    else:
        # Find all valid (H, W) where both are powers of 2 and H * W = N
        candidates = []
        for i in range(1, int(math.log2(N))):
            H_candidate = 2 ** i
            W_candidate = N // H_candidate
            if (W_candidate & (W_candidate - 1)) == 0:  # W is power of 2
                candidates.append((H_candidate, W_candidate))

        if not candidates:
            raise ValueError("No valid (H, W) found")

        # Pick the most balanced pair (closest to square)
        H, W = min(candidates, key=lambda x: abs(x[0] - x[1]))

    return x.view(B, H, W, C)


def scene_similarity_heatmaps(
    tokens: torch.Tensor,
    logger,
    step: int,
    ref_scene: int=0,
    ref_token_idx: int=0, 
    save_path: Optional[Path | str] = None,
    name: Optional[str] = None,
):
    """
    tokens:         [B, N, C]
    ref_scene:      index b0
    ref_token_idx:  index n0 within scene b0
    H_p, W_p:       grid dims so that N = H_p * W_p
    """
    B, N, C = tokens.shape
    try:
        ref_vec = tokens[ref_scene, ref_token_idx].float().unsqueeze(0)   # [1, C]
        all_flat = tokens.float().view(B * N, C)                          # [B*N, C]
        sims = F.cosine_similarity(ref_vec, all_flat, dim=1)      # [B*N]
        sims = sims.view(B, N)                      # [B, N]

        sims = reshape_tokens_to_image(sims[..., None])[..., 0].cpu().numpy()
        b, H_p, W_p = sims.shape
        sims_map = sims.reshape(B, H_p, W_p)                       # [B, H_p, W_p]

        fig, axes = plt.subplots(1, B, figsize=(3*B, 3))
        canvas = FigureCanvas(fig)
        for b in range(B):
            ax = axes[b] if B > 1 else axes
            im = ax.imshow(sims_map[b], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Scene {b} sim to (scene{ref_scene},tok{ref_token_idx})")
            ax.axis("off")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.set_label("Cosine Similarity")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # Render canvas and convert to NumPy array
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        plt.close(fig)
        
        logger.log_image(
            f"sim/heatmap" if name is None else f"sim/heatmap/{name}",
            [image],
            step=step,
            caption=[f"Cross Scene Token Similarity"],
        )
    except Exception as err:
        print(err)
    
def plot_scene_wise_pca(
    tokens: torch.Tensor, 
    names: list[str], 
    logger,
    step: int, 
    n_components: int = 2, 
    save_path: Optional[Path | str] = None
):
    """
    For each scene, do a PCA on its N tokens → plot the top 2 components.
    tokens: [B, N, C]
    """
    B, N, C = tokens.shape
    try:
        cmap = cm.get_cmap('tab10' if B <= 10 else 'tab20', B)
        fig, axes = plt.subplots(1, B, figsize=(4*B, 4))

        canvas = FigureCanvas(fig)
        for b in range(B):
            # Compute PCA for scene b
            pca = PCA(n_components=n_components)
            z_b = tokens[b].float().cpu().numpy()          # [N, C]
            coords = pca.fit_transform(z_b)         # [N, 2]
            
            ax = axes[b] if B > 1 else axes
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=12,
                alpha=0.6,
                color=cmap(b)                       # use distinct color for scene b
            )
            ax.set_title(f"PCA of Scene Tokens {names[b]}", color=cmap(b))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            # ax.axis("off")
        
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # Render canvas and convert to NumPy array
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        plt.close(fig)
        
        logger.log_image(
            f"pca/local",
            [image],
            step=step,
            caption=[f"PCA per Scene"],
        )
    except Exception as err:
        print(err)

def plot_global_pca(
    tokens: torch.Tensor, 
    logger,
    step: int, 
    n_components: int = 2,
    name: Optional[str] = None,
):
    """
    Compute PCA over all tokens from all scenes, then plot the first two principal components
    in a single scatter, coloring points by scene index.

    Args:
        tokens:        torch.Tensor of shape [B, N, C]
        n_components:  number of PCA components (must be >=2 to plot 2D)
    """
    B, N, C = tokens.shape
    try:
        # 1) Flatten to [B*N, C]
        all_flat = tokens.reshape(B * N, C).float().cpu().numpy()

        # 2) (Optional) Normalize each token to unit length
        norms = np.linalg.norm(all_flat, axis=1, keepdims=True) + 1e-12
        all_flat = all_flat / norms

        # 3) Run PCA on all [B*N, C]
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(all_flat)  # shape = [B*N, 2]

        # 4) Build scene labels for each point
        labels = np.repeat(np.arange(B), N)   # shape = [B*N]

        # 5) Choose a categorical colormap with B distinct colors
        cmap = cm.get_cmap('tab10' if B <= 10 else 'tab20', B)

        # 6) Plot all points in one figure, colored by scene index
        fig, ax = plt.subplots(figsize=(5, 5))
        canvas = FigureCanvas(fig)

        for b in range(B):
            idxs = np.where(labels == b)[0]
            ax.scatter(
                coords[idxs, 0],
                coords[idxs, 1],
                s=12,
                alpha=0.6,
                color=cmap(b),
                label=f"Scene {b}"
            )
        ax.legend(loc='best', fontsize='small', ncol=2)
        ax.set_title("Global PCA of All Tokens (color = scene)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        plt.tight_layout()
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        plt.close(fig)
        
        logger.log_image(
            f"pca/global" if name is None else f"pca/global/{name}",
            [image],
            step=step,
            caption=[f"PCA Global"],
        )
    except Exception as err:
        print(err)

def plot_tsne_to_numpy(
    tensor, 
    names: list[str], 
    logger,
    step: int,
    save_path: Optional[Path | str] = None,
    name: Optional[str] = None,
):
    """
    Generates a t-SNE plot from a [B, N, C] tensor and returns it as a NumPy image (H, W, 3).
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [B, N, C]
    
    Returns:
        np.ndarray: RGB image of the t-SNE plot (dtype=np.uint8, shape [H, W, 3])
    """
    B, N, C = tensor.shape
    try:
        # Flatten to [B*N, C]
        data = tensor.reshape(B * N, C).float().cpu().numpy()

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(data)

        # Create figure and canvas
        fig, ax = plt.subplots(figsize=(5, 5))
        canvas = FigureCanvas(fig)

        # Assign colors per batch
        cmap = cm.get_cmap('tab10' if B <= 10 else 'tab20', B)

        for b in range(B):
            indices = slice(b * N, (b + 1) * N)
            ax.scatter(
                tsne_result[indices, 0],
                tsne_result[indices, 1],
                color=cmap(b),
                label=f'Scene {names[b]}',
                alpha=0.7,
                s=30
            )

        ax.legend()
        ax.set_title("t-SNE plot colored by batch")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(True)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # Render canvas and convert to NumPy array
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        plt.close(fig)
        
        logger.log_image(
            f"tsne/2d" if name is None else f"tsne/2d/{name}",
            [image],
            step=step,
            caption=[f"TSNE-2D Plot"],
        )
    except Exception as err:
        print(err)

def plot_tsne3d_to_numpy(
    tensor, 
    names: list[str], 
    logger,
    step: int,
    save_path: Optional[Path | str] = None,
):
    """
    Generates a 3D t-SNE plot from a [B, N, C] tensor and returns it as a NumPy RGB image.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [B, N, C]
    
    Returns:
        np.ndarray: RGB image of the 3D t-SNE plot (dtype=uint8, shape [H, W, 3])
    """
    B, N, C = tensor.shape
    try:
        data = tensor.reshape(B * N, C).float().cpu().numpy()

        # 3D t-SNE
        tsne = TSNE(n_components=3, random_state=42)
        tsne_result = tsne.fit_transform(data)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        canvas = FigureCanvas(fig)

        cmap = cm.get_cmap('tab10' if B <= 10 else 'tab20', B)

        for b in range(B):
            idx = slice(b * N, (b + 1) * N)
            ax.scatter(
                tsne_result[idx, 0],
                tsne_result[idx, 1],
                tsne_result[idx, 2],
                color=cmap(b),
                label=f'Scene {names[b]}',
                alpha=0.7,
                s=30
            )

        ax.set_title("3D t-SNE plot colored by batch")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Render the canvas and convert to numpy
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        plt.close(fig)
        
        logger.log_image(
            f"tsne/3d",
            [image],
            step=step,
            caption=[f"TSNE-3D Plot"],
        )
    except Exception as err:
        print(err)



def plot_variance_colored_bars(data, path):
    """
    Args:
        data (np.ndarray): A NumPy array of shape (B, N, C)
    """

    if data.ndim != 3:
        raise ValueError("Input must be a 3D array of shape (B, N, C)")

    B, N, C = data.shape

    # Compute variance across B (shape: N, C), then average across channels -> shape: (N,)
    var_across_B = np.var(data, axis=0)  # shape (N, C)
    var_final = np.mean(var_across_B, axis=1)  # shape (N,)

    # Normalize variance for coloring
    norm = colors.Normalize(vmin=np.min(var_final), vmax=np.max(var_final))
    cmap = cm.viridis
    bar_colors = cmap(norm(var_final))

    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=(30, 6))
    x = np.arange(N)
    bar_height = 1  # constant height

    # Plot bars with color mapped to variance
    bars = ax.bar(x, [bar_height] * N, color=bar_colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Vector index (N)')
    ax.set_yticks([])
    ax.set_title('Variance across B (averaged over C) shown as bar color')

    # Create and attach colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(var_final)  # this links the colorbar to our data
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.25)
    cbar.set_label('Variance')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()