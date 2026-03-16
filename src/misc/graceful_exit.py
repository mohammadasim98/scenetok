"""
Graceful Exit Callback for PyTorch Lightning

Handles SIGTERM signals (e.g., from SLURM job preemption) to:
1. Save a checkpoint atomically (avoids corruption)
2. Gracefully stop the training loop

Usage:
    Add `GracefulExitCallback()` to your Trainer callbacks.
    Works with SLURM's `#SBATCH --signal=B:SIGTERM@120` directive.
"""

import os
import signal
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info


class GracefulExitCallback(Callback):
    """
    Callback that handles SIGTERM for graceful shutdown with checkpoint saving.
    
    On receiving SIGTERM:
    1. Saves checkpoint atomically (write to temp, then rename)
    2. Sets trainer.should_stop = True to exit after current batch
    
    Args:
        checkpoint_dir: Directory to save the checkpoint. If None, uses trainer's default.
        checkpoint_name: Name of the checkpoint file. Default: "last.ckpt"
    """
    
    def __init__(
        self, 
        checkpoint_dir: Optional[Path] = None,
        checkpoint_name: str = "last.ckpt"
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self._trainer: Optional[Trainer] = None
        self._received_sigterm = False
        self._original_handler = None
    
    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM signal."""
        if self._received_sigterm:
            # Already handling, ignore duplicate signals
            return
        
        self._received_sigterm = True
        rank_zero_info("\n" + "="*60)
        rank_zero_info("SIGTERM received - initiating graceful shutdown...")
        rank_zero_info("="*60)
        
        if self._trainer is not None:
            # Signal the trainer to stop after current batch
            self._trainer.should_stop = True
            
            # Save checkpoint immediately (atomic write)
            self._save_checkpoint_atomic()
    
    @rank_zero_only
    def _save_checkpoint_atomic(self):
        """Save checkpoint atomically to prevent corruption."""
        if self._trainer is None:
            return
        
        # Determine checkpoint directory
        if self.checkpoint_dir is not None:
            ckpt_dir = Path(self.checkpoint_dir)
        elif self._trainer.default_root_dir:
            ckpt_dir = Path(self._trainer.default_root_dir) / "checkpoints"
        else:
            ckpt_dir = Path("checkpoints")
        
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = ckpt_dir / self.checkpoint_name
        
        # Write to temporary file first, then atomic rename
        # This prevents corruption if the process is killed during write
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".ckpt.tmp",
            dir=str(ckpt_dir)
        )
        os.close(temp_fd)
        
        try:
            rank_zero_info(f"Saving checkpoint to {final_path}...")
            self._trainer.save_checkpoint(temp_path)
            
            # Atomic rename (on same filesystem, rename is atomic)
            shutil.move(temp_path, final_path)
            
            rank_zero_info(f"Checkpoint saved successfully at step {self._trainer.global_step}")
            rank_zero_info("="*60 + "\n")
            
        except Exception as e:
            rank_zero_info(f"Error saving checkpoint: {e}")
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Register SIGTERM handler when training starts."""
        self._trainer = trainer
        self._received_sigterm = False
        
        # Store original handler and install ours
        self._original_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        rank_zero_info("GracefulExitCallback: SIGTERM handler registered")
    
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        """Restore original signal handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGTERM, self._original_handler)
            self._original_handler = None
        
        if self._received_sigterm:
            rank_zero_info("Training ended gracefully after SIGTERM")
    
    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: Exception):
        """Handle exceptions - save checkpoint if SIGTERM was received."""
        if self._received_sigterm and not isinstance(exception, SystemExit):
            rank_zero_info(f"Exception during graceful shutdown: {exception}")
