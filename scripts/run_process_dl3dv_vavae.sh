#!/bin/bash

# Job constants
PARTITION="gpu20"  # GPU Partition
TIME="00-01:00:00"
CPUS="10"


size=$1
stage=$2
tasks=$3
flip=$4
output_dir=$5
root=$6
subset=$7

config="va_dl3dv"

jobscript="jobs/${config}.sh"
output="job_outputs/${config}.o%A_%a"
mkdir -p "$(dirname "${jobscript}")"
mkdir -p "$(dirname "${output}")"
echo "#!/bin/bash" > $jobscript



### Partition name
echo "#SBATCH -p ${PARTITION}" >> $jobscript
# CPU
echo "#SBATCH --cpus-per-task=${CPUS}" >> $jobscript
### Job name
echo "#SBATCH --job-name ${config}" >> $jobscript
### File for the output
echo "#SBATCH --output ${output}" >> $jobscript
### Time your job needs to execute, e.g. 15 min 30 sec
echo "#SBATCH --time ${TIME}" >> $jobscript
### Array execution
echo "#SBATCH -a 0-${tasks}%20" >> $jobscript
### Start time for delayed execution
echo "#SBATCH --mem=125G" >> $jobscript

### Number of GPUs per node, I want to use, e.g. 1
echo "#SBATCH --gres gpu:1" >> $jobscript

echo "echo -n 'date: ';(date '+%Y-%m-%d %H:%M:%S')" >> $jobscript
# echo "unset LOCAL_RANK" >>  $jobscript

echo "conda activate scenetok" >>  $jobscript # <-- Change if using another environment manager
echo 'echo "${SLURM_ARRAY_TASK_ID}"' >> $jobscript
echo 'cd "${PROJECT_ROOT}"' >> $jobscript

echo 'python -m src.scripts.preprocess_dataset dataset=va_dl3dv dataset.data_root='"${root}"' stage='"${stage}"' dataset.subset='"${subset}"' output_dir='"${output_dir}"' flip='"${flip}"' index="${SLURM_ARRAY_TASK_ID}" size='"${size}"' dataset.num_workers='"${CPUS}">> $jobscript

echo $jobscript
sbatch $jobscript
# rm $jobscript