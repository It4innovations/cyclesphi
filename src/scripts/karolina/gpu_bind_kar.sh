#!/bin/bash

#export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
# export CUDA_VISIBLE_DEVICES=$(( $PMIX_RANK % 8 ))
# #echo $CUDA_VISIBLE_DEVICES

# if [ $PMIX_RANK -eq 0 ]
# then
#     export DISPLAY=:0.0
#     echo  $PMIX_RANK $DISPLAY $CUDA_VISIBLE_DEVICES
#     $@
# else
#     export DISPLAY=:0.0
#     echo  $PMIX_RANK $DISPLAY $CUDA_VISIBLE_DEVICES
#     $@
# fi

# Get local rank using SLURM_LOCALID (most direct method)
LOCAL_RANK=$SLURM_LOCALID

# Alternative if SLURM_LOCALID isn't available
if [ -z "$LOCAL_RANK" ]; then
  # Calculate using node ID and local task ID
  LOCAL_RANK=$((SLURM_PROCID - SLURM_NODEID * SLURM_NTASKS_PER_NODE))
fi

export CUDA_VISIBLE_DEVICES=$(( $LOCAL_RANK % 8 )) #$LOCAL_RANK
echo "Process $SLURM_PROCID: Local rank $LOCAL_RANK on node $SLURM_NODEID, Using GPU: $CUDA_VISIBLE_DEVICES"
#echo  $PMIX_RANK $CUDA_VISIBLE_DEVICES
$@