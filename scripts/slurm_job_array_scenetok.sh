#!/bin/bash

# Job constants
PARTITION="gpu24"   # GPU Partition
TASK_STEPS="30000" # Number of tasks per step (once the steps are reached, task will terminate)
TIME="00-02:15:00"  # NOTE make sure that TASK_STEPS take less than this time!
TASKS="100"
MEMORY="128G" # Memory e.g. 128G
GPUS="4" # Number of GPUs e.g., 4   
CPUS="30" # Number of CPUS per task
NUM_NODES="1" # Number of nodes

config=$1
id=${2:-'null'} # If not provided, then select a date as the unique id

if [ "$id" == 'null' ]; then
    id=$(date '+%Y-%m-%d_%H-%M-%S')
fi

jobscript="jobs/${config}.sh"
output="job_outputs/${config}.o%A_%a"
mkdir -p "$(dirname "${jobscript}")"
mkdir -p "$(dirname "${output}")"
echo "#!/bin/bash" > $jobscript



### Partition name
echo "#SBATCH -p ${PARTITION}" >> $jobscript
echo "#SBATCH --nodes=${NUM_NODES}" >> $jobscript
echo "#SBATCH --ntasks-per-node=${GPUS}" >> $jobscript

# Define SIGTERM
echo "#SBATCH --signal=B:SIGTERM@120" >> $jobscript
# CPU
echo "#SBATCH --cpus-per-task=${CPUS}" >> $jobscript
### Job name
echo "#SBATCH --job-name ${config}" >> $jobscript
### File for the output
echo "#SBATCH --output ${output}" >> $jobscript
### Time your job needs to execute, e.g. 15 min 30 sec
echo "#SBATCH --time ${TIME}" >> $jobscript
### Array execution
echo "#SBATCH -a 1-${TASKS}%1" >> $jobscript

### Number of GPUs per node, I want to use, e.g. 1
echo "#SBATCH --gres gpu:${GPUS}" >> $jobscript

echo "echo -n 'date: ';(date '+%Y-%m-%d %H:%M:%S')" >> $jobscript
echo 'trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT' >> $jobscript

echo "conda activate scenetok" >>  $jobscript # <-- Change if using another environment manager
echo 'cd "${PROJECT_ROOT}"' >> $jobscript

echo "srun --mem-per-gpu=${MEMORY} python -m src.main +experiment=${config} data_loader.train.num_workers=${CPUS} hydra.run.dir=./outputs/${config}/${id} mode=train hydra.job.name=train trainer.task_steps=${TASK_STEPS} trainer.devices=${GPUS} trainer.num_nodes=${NUM_NODES} &" >> $jobscript
echo "wait" >> $jobscript

echo $jobscript
sbatch $jobscript


