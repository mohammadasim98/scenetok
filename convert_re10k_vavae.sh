#!/bin/bash

# VideoDC
output_dir=$1
root_dir=$2
size=20
tasks=300

bash scripts/run_process_re10k_vavae.sh ${size} train ${tasks} false ${output_dir} ${root_dir}
bash scripts/run_process_re10k_vavae.sh ${size} train ${tasks} true ${output_dir} ${root_dir}

bash scripts/run_process_re10k_vavae.sh ${size} test ${tasks} false ${output_dir} ${root_dir}

