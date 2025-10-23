#!/bin/bash

cd /mnt/proj3/open-28-64/blender/salomon/projects/blender

#module use /mnt/proj3/open-18-15/blender/vistle/easybuild/modules/all
#ml tbb
#ml tbb/2020.1-GCCcore-9.3.0

ml CMake/3.24.3-GCCcore-12.2.0
#ml Mesa #/22.0.3-GCCcore-11.3.0
ml Mesa/22.2.4-GCCcore-12.2.0

ml OpenMPI/4.1.6-NVHPC-23.11-CUDA-12.2.0
#ml OpenMPI/4.1.4-GCC-12.2.0
ml GCC/12.2.0

#ml VirtualGL

ROOT_DIR=${PWD}

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/cycles_kar_gpu
src=${ROOT_DIR}/src

# CYCLES_DUMPGRAPH_MATNAME_=M_Galaxy_1
# CYCLES_DUMPGRAPH_PATH_=e:\tmp\cycles_graphs\
#OPTIX_ROOT_DIR=C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0

#export CYCLES_XML_PATH=${ROOT_DIR}/data/xml/scene.xml
export CYCLES_XML_PATH=${ROOT_DIR}/data/xml/scene_gadget.xml
#export CYCLES_XML_PATH=${ROOT_DIR}/data/xml/lone-monk.xml
#export CYCLES_XML_PATH=${ROOT_DIR}/data/xml/scene_space.xml

export LD_LIBRARY_PATH=$lib_dir/gpujpeg_kar_gpu/lib:$LD_LIBRARY_PATH

#CYCLESPHI_BMP=e:/tmp/cyclesphi
export CYCLESPHI_USE_GPU=OPTIX
# OPTIX_ROOT_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.3.0
# CYCLES_XML_PATH=e:\tmp\cycles_graphs\volume_scene.xml
# CYCLES_XML_PATH_=f:\work\blender\cycles\examples\scene_cube_surface.xml
# OCIO=f:\work\blender\blender\release\datafiles\colormanagement\config.ocio
export CYCLES_VOLUME_GEOM=VolumeGeom
export CYCLES_VOLUME_ATTR=density

# Select which GPU to use (first GPU)
#export CUDA_VISIBLE_DEVICES=0

#export CLIENT_IN_UNIMEM=none #nanovdb

#${output}/cyclesphi_space --port 8000 --space-port 6000 --space-server cn266 --space-server-port 5000
#--scene e:\tmp\cycles_graphs\falcon\anim\falcon --device OPTIX --anim 2
#/scratch/project/open-31-44/it4i-adriank/olb/apps/adrian/fsi/airplane3d/test_output_1
#/scratch/project/open-31-44/it4i-adriank/olb/apps/adrian/fsi/airplane3d/test_output_1/volumes/volume_00361.vdb
srun -u -n 16 -c 4 ${ROOT_DIR}/scripts/gpu_bind_kar.sh ${output}/cyclesphi_mpi --port 8000 --scene /scratch/project/open-28-64/milanjaros/anim/scene --device OPTIX #--anim 4
