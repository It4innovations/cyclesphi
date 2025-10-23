#!/bin/bash

#qsub -I -l select=1 -l filesystems=flare -l walltime=1:00:00 -q debug -A Insitu
#ml oneapi/release/2024.2.1

cd ~/scratch/projects/cycles

ROOT_DIR=${PWD}

#ml oneapi/release/2024.2.1
#ml oneapi/eng-compiler/2024.07.30.002
#ml mpich
module restore
#ml
ml cmake
##################################################

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/cycles_aurora_gpu
src=${ROOT_DIR}/src

#export ONEAPI_DEVICE_SELECTOR=level_zero:0  # First GPU, first tile

mpirun -n 2 ${ROOT_DIR}/scripts/gpu_bind.sh ${output}/cyclesphi_mpi --port 8001 --scene ${ROOT_DIR}/data/fal/_lfs/scene --device ONEAPI #--anim 4