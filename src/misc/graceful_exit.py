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
import sys
import signal
from pathlib import Path
from typing import Optional

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback


def _print_flush(msg: str):
    """Print with immediate flush for signal handlers."""
    print(msg, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
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
        
        # Print on all ranks so we know signal was received
        _print_flush("\n" + "="*60)
        _print_flush(f"[Rank {self._trainer.global_rank if self._trainer else '?'}] SIGTERM received - initiating graceful shutdown...")
        _print_flush("="*60)
        
        if self._trainer is not None:
            # Signal the trainer to stop after current batch
            self._trainer.should_stop = True
            
            # Save checkpoint immediately (atomic write)
            self._save_checkpoint_atomic()
    
    def _save_checkpoint_atomic(self):
        """Save checkpoint atomically to prevent corruption."""
        if self._trainer is None:
            return
        
        # Only rank 0 should save
        if self._trainer.global_rank != 0:
            return
        
        # Use the checkpoint_dir passed during init
        if self.checkpoint_dir is not None:
            ckpt_dir = self.checkpoint_dir
        else:
            ckpt_dir = Path(self._trainer.default_root_dir) / "checkpoints"
        
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = ckpt_dir / self.checkpoint_name
        temp_path = ckpt_dir / f".{self.checkpoint_name}.tmp"
        
        try:
            _print_flush(f"[Rank 0] Saving checkpoint to {final_path}...")
            
            # Save to temp file first
            self._trainer.save_checkpoint(str(temp_path))
            
            # Atomic rename (on same filesystem, rename is atomic on POSIX)
            os.replace(str(temp_path), str(final_path))
            
            _print_flush(f"[Rank 0] Checkpoint saved successfully at step {self._trainer.global_step}")
            _print_flush("="*60 + "\n")
            
        except Exception as e:
            _print_flush(f"[Rank 0] Error saving checkpoint: {e}")
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Register SIGTERM handler when training starts."""
        self._trainer = trainer
        self._received_sigterm = False
        
        # Store original handler and install ours
        self._original_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        if trainer.global_rank == 0:
            _print_flush(f"GracefulExitCallback: SIGTERM handler registered, checkpoint_dir={self.checkpoint_dir}")
    
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        """Restore original signal handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGTERM, self._original_handler)
            self._original_handler = None
        
        if self._received_sigterm and trainer.global_rank == 0:
            _print_flush("Training ended gracefully after SIGTERM")
    
    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: Exception):
        """Handle exceptions - save checkpoint if SIGTERM was received."""
        if self._received_sigterm and not isinstance(exception, SystemExit):
            _print_flush(f"Exception during graceful shutdown: {exception}")
