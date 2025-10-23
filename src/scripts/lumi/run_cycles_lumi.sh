#!/bin/bash

#salloc -A project_465001729 --partition=standard-g -N 1 -n 8 -t 04:00:00 --gres=gpu:8

cd ~/scratch/blender/

ROOT_DIR=${PWD}

# ml CMake/3.24.3-GCCcore-12.2.0
# ml Mesa/22.2.4-GCCcore-12.2.0

# ml OpenMPI/4.1.6-NVHPC-23.11-CUDA-12.2.0
# ml GCC/12.2.0

ml LUMI/24.03  partition/G
ml rocm/6.2.2
ml gcc
##################################################

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/cycles_lumi_gpu
src=${ROOT_DIR}/src

export HIP_VISIBLE_DEVICES=0
#/pfs/lustrep2/scratch/project_465001561/jaromila/blender/scripts/gpu_bind.sh
srun -n 2 /pfs/lustrep2/scratch/project_465001561/jaromila/blender/scripts/gpu_bind.sh ${output}/cyclesphi_mpi --port 8003 --scene /users/jaromila/scratch/blender/data/fal/scene --device HIP #--anim 4