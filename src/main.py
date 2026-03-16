




import os
import hydra
import wandb
import warnings

from pathlib import Path
from torch import manual_seed, load
from colorama import Fore
from omegaconf import DictConfig, OmegaConf

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.misc.graceful_exit import GracefulExitCallback
from src.model.diffusion_wrapper import DiffusionWrapper
from src.profiler import get_profiler

warnings.simplefilter(action='ignore', category=FutureWarning)

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    # print(cfg_dict.dataset)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    if cfg_dict.seed is not None:
        manual_seed(cfg_dict.seed)
    # Set up the output directory.

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )    
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.activated:
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{output_dir.parent.name} ({output_dir.name})",
            id=f"{output_dir.parent.name}_{output_dir.name}",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            entity=cfg_dict.wandb.entity
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    checkpoint_dir = output_dir / "checkpoints"
    if cfg.checkpointing.save:
        if cfg.checkpointing.every_n_train_steps is not None:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_dir,
                    every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                    save_top_k=cfg.checkpointing.save_top_k,
                    save_last=True,
                    save_on_train_epoch_end=False,
                    verbose=True,
                    enable_version_counter=False
                )
            )
        # Add graceful exit handler for SLURM preemption signals
        callbacks.append(GracefulExitCallback(checkpoint_dir=checkpoint_dir))

    # Prepare the checkpoint for loading.
    checkpoint_path = checkpoint_dir / "last.ckpt"
    if os.path.exists(checkpoint_path):
        resume = True
        print("(Main) Resuming from: ", checkpoint_path)
    else:
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
        resume = cfg.checkpointing.resume

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker(cfg.train.step_offset)

    
    kwargs = dict(
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        freeze_cfg = cfg.freeze,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        val_cfg=cfg.val,
        sampler_cfg=cfg.sampler,
        step_tracker=step_tracker,
        output_dir=output_dir,
        batch_size=cfg.data_loader.train.batch_size,
        val_check_interval=cfg.trainer.val_check_interval,
        mode=cfg.mode,
    )
    if cfg.mode == "train" and checkpoint_path is not None and not resume:
        # Just load model weights but no optimizer state
        model_wrapper = DiffusionWrapper.load_from_checkpoint(checkpoint_path, **kwargs)
    else:
        model_wrapper = DiffusionWrapper(**kwargs)
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)



    step = load(checkpoint_path, "cpu", weights_only=False)["global_step"] if cfg.mode == "train" and checkpoint_path is not None else 0
    max_steps = cfg.trainer.max_steps if cfg.trainer.task_steps is None else min(step+cfg.trainer.task_steps, cfg.trainer.max_steps)

    val_dataloaders = data_module.val_dataloader()

    print("Number of nodes: ", cfg.trainer.num_nodes)
    print("Number of devices/node: ", cfg.trainer.devices)
    print("Number of context: ", cfg.dataset.view_sampler.num_context_views)
    print("Number of target: ", cfg.dataset.view_sampler.num_target_views)
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        limit_val_batches=1 if cfg.trainer.validate else 0,
        val_check_interval=cfg.trainer.val_check_interval if cfg.trainer.validate else None,
        check_val_every_n_epoch=None,
        enable_checkpointing=cfg.checkpointing.save,
        default_root_dir=output_dir,
        enable_progress_bar=True if "predict" in cfg.mode else False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=max_steps,
        strategy=cfg.trainer.strategy,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        profiler=get_profiler(cfg.trainer.profiler)
    )
    # model_wrapper.strict_loading = True
    if cfg.mode == "train":
        trainer.fit(model_wrapper, train_dataloaders=data_module.train_dataloader(), val_dataloaders=val_dataloaders["standard"], ckpt_path=checkpoint_path if resume else None)
    elif cfg.mode == "val":
        trainer.validate(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    elif cfg.mode == "test":
        trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    elif cfg.mode == "predict_test":
        trainer.predict(model_wrapper, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)
    elif cfg.mode == "predict_train":
        trainer.predict(model_wrapper, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    train()
