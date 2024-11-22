#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J GPU_train
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
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
#load enviroment
source "${BLACKHOLE}/DL/bin/activate"

python LoRA.py





