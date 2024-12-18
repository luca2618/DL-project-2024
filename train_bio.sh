#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J GPU_train_bio
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./log/gpu_%J.out
#BSUB -e ./log/gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
#export HF_TOKEN_PATH="/dtu/blackhole/00/167776/DL-project-2024/cache/token"
export HF_HOME="/dtu/blackhole/00/167776/cache"
#load enviroment
source "${BLACKHOLE}/DL/bin/activate"

model=Mistral-7B-Instruct-v0.1
dataset=Bio

python LoRA.py "$model" "$dataset"




