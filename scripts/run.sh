#!/bin/bash

# Job constants

config=$1
num_workers=${2:-16}
gpus=${3:-1}
id=${4:-'null'}
params=${*:3}

if [ "$id" == 'null' ]; then
    id=$(date '+%Y-%m-%d_%H-%M-%S')
fi
python -m src.main +experiment=${config} data_loader.train.num_workers=${num_workers} trainer.val_check_interval=2000 data_loader.val.standard.batch_size=8 hydra.run.dir=./outputs/${config}/${id} mode=train hydra.job.name=train checkpointing.every_n_train_steps=1000 trainer.devices=${gpus} trainer.num_nodes=1 wandb.activated=true
