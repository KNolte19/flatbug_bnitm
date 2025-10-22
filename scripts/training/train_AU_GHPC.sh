#!/bin/bash

# # SBATCH -p ghpc_gpu                 # Name of the queue
# # SBATCH -N 1                       # Number of nodes(DO NOT CHANGE)
# # SBATCH -n 16                       # Number of CPU cores
# # SBATCH --mem=64000                 # Memory in MiB(10 GiB = 10 * 1024 MiB)
# # SBATCH -t 96:00:00 

CONFIG=fb_config_M40S_GHPC.yaml
#ROOT=/home/altair/flat-bug
# source ${ROOT}/.venv/bin/activate
fb_clone_data -s secrets.yaml -o ${ROOT}/flat-bug-data/pre-pro/
fb_prepare_data -i ${ROOT}/flat-bug-data/pre-pro/  -o ${ROOT}/flat-bug-data/yolo/ -f
fb_train -c ${ROOT}/scripts/training/${CONFIG} -d ${ROOT}/flat-bug-data/yolo/

