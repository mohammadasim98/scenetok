#!/bin/bash

# VideoDC
output_dir=$1
root_dir=$2
size=10
tasks=100

bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 1K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 2K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 3K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 4K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 5K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 6K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 7K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 8K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 9K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 10K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir} 11K

bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 1K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 2K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 3K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 4K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 5K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 6K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 7K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 8K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 9K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 10K
bash scripts/run_process_dl3dv_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir} 11K

bash scripts/run_process_dl3dv_vavae.sh 10 test 20 false ${output_dir} ${root_dir}
