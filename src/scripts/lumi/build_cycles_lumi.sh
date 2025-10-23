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


export CC=gcc
export CXX=g++

#export CUDA_ROOT=/usr/local/cuda
#echo $CUDA_ROOT
#export OptiX_INSTALL_DIR=$lib_dir/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64
#export OptiX_INSTALL_DIR=$lib_dir/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64
#export LD_LIBRARY_PATH=${OptiX_INSTALL_DIR}:${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH
#export PATH=${CUDA_ROOT}/bin:$PATH

# rm -rf ${ROOT_DIR}/build/cycles_lumi_gpu
# rm -rf ${ROOT_DIR}/install/cycles_lumi_gpu

#-----------blender_client--------------
mkdir ${ROOT_DIR}/build/cycles_lumi_gpu
cd ${ROOT_DIR}/build/cycles_lumi_gpu

make_d="${src}/cycles"

make_d="${make_d} -DCMAKE_BUILD_TYPE=RelWithDebInfo"
#make_d="${make_d} -DCMAKE_BUILD_TYPE=Release"
#make_d="${make_d} -DCMAKE_BUILD_TYPE=Debug"

make_d="${make_d} -DCMAKE_INSTALL_PREFIX=${output}"
#make_d="${make_d} -DUMESH_ENABLE_SANITY_CHECKS=ON"
#make_d="${make_d} -DBUILD_VIEWER=OFF"

make_d="${make_d} -DWITH_CYCLES_HYDRA_RENDER_DELEGATE=OFF"
make_d="${make_d} -DWITH_CYCLES_USD=OFF"

make_d="${make_d} -DWITH_CLIENT_MPI_SOCKET=ON"

#make_d="${make_d} -DOPTIX_ROOT_DIR=$OptiX_INSTALL_DIR"
#make_d="${make_d} -DCMAKE_CUDA_ARCHITECTURES=80"
#make_d="${make_d} -DWITH_CYCLES_CUDA_BINARIES=ON"
make_d="${make_d} -DWITH_CYCLES_DEVICE_HIP=ON"
make_d="${make_d} -DWITH_CYCLES_HIP_BINARIES=ON"
#make_d="${make_d} -DCYCLES_CUDA_BINARIES_ARCH=sm_80"
make_d="${make_d} -DCYCLES_HIP_BINARIES_ARCH=gfx90a"
#make_d="${make_d} -DCUDA_BINARIES_ARCH:STRING=sm_80"
#make_d="${make_d} -DWITH_CUDA_DYNLOAD=OFF"
make_d="${make_d} -DWITH_CYCLES_LOGGING=OFF"
make_d="${make_d} -DWITH_GPU_CPUIMAGE=ON"

make_d="${make_d} -DGLOG_INCLUDE_DIR=/users/jaromila/scratch/blender/src/glog/install/include"
make_d="${make_d} -DGLOG_LIBRARY=/users/jaromila/scratch/blender/src/glog/install/lib64/libglog.so"

make_d="${make_d} -DGFLAGS_INCLUDE_DIR=/users/jaromila/scratch/blender/src/gflags/install/include"
make_d="${make_d} -DGFLAGS_LIBRARY=/users/jaromila/scratch/blender/src/gflags/install/lib/libgflags.so"

#make_d="${make_d} -Danari_DIR=$lib_dir/anari-sdk_kar_gpu/lib64/cmake/anari-0.10.1"

# make_d="${make_d} -DOWL_LIBRARIES="
# make_d="${make_d} -DOWL_INCLUDES=$src/rtxumeshviewer/submodules/owl/owl/include"
# make_d="${make_d} -DOWL_INCLUDES2=$src/rtxumeshviewer/submodules/owl"
# make_d="${make_d} -DICET_INCLUDES=$src/rtxumeshviewer/submodules/icet/install/include"
# make_d="${make_d} -DUMESH_INCLUDES=$src/rtxumeshviewer/submodules/umesh"
# make_d="${make_d} -DOPTIX_INCLUDES=$OptiX_INSTALL_DIR/include"
# make_d="${make_d} -DGLFW_INCLUDES=$src/rtxumeshviewer/submodules/owl/3rdParty/glfw/include"
#make_d="${make_d} -Dglfw3_DIR=/mnt/proj3/open-28-64/blender/salomon/projects/ingo/install/glfw/lib64/cmake/glfw3"

# make_d="${make_d} -DUMESH_LIBRARIES=$src/rtxumeshviewer/submodules/umesh/build/libumesh.a;/apps/all/tbb/2020.3-GCCcore-10.3.0/lib/libtbb.so;/apps/all/tbb/2020.3-GCCcore-10.3.0/lib/libtbbmalloc.so"
# make_d="${make_d} -DOWL_LIBRARIES=$src/rtxumeshviewer/submodules/owl/build/libowl.a;${CUDA_ROOT}/lib64/stubs/libcuda.so;${CUDA_ROOT}/lib64/libcudart.so"
# make_d="${make_d} -DOWL_VIEWER_LIBRARIES=$src/rtxumeshviewer/submodules/owl/build/libowl_viewer.a;$src/rtxumeshviewer/submodules/owl/build/libglfw3.a;GL;X11"

# make_d="${make_d} -DQT_OWL_INCLUDES=$src/rtxumeshviewer/submodules/cuteeOWL;/apps/all/Qt5/5.15.5-GCCcore-11.3.0/include/QtGui;/apps/all/Qt5/5.15.5-GCCcore-11.3.0/include/QtWidgets;/apps/all/Qt5/5.15.5-GCCcore-11.3.0/include/QtCore"
# make_d="${make_d} -DQT_OWL_LIBRARIES=$src/rtxumeshviewer/submodules/cuteeOWL/build/qtOWL/libqtOWL.a;${CUDA_ROOT}/lib64/stubs/libcuda.so;${CUDA_ROOT}/lib64/libcudart.so;/apps/all/Qt5/5.15.5-GCCcore-11.3.0/lib/libQt5Gui.so;/apps/all/Qt5/5.15.5-GCCcore-11.3.0/lib/libQt5Widgets.so;/apps/all/Qt5/5.15.5-GCCcore-11.3.0/lib/libQt5Core.so"

#make_d="${make_d} -DUMESH_LIBRARIES=/apps/all/tbb/2020.3-GCCcore-10.3.0/lib/libtbb.so;/apps/all/tbb/2020.3-GCCcore-10.3.0/lib/libtbbmalloc.so"
#make_d="${make_d} -DUMESH_USE_TBB=OFF"

#make_d="${make_d} -DWITH_CLIENT_GPUJPEG=ON"
#make_d="${make_d} -DGPUJPEG_INCLUDE_DIR=$lib_dir/gpujpeg_kar_gpu/include"
#make_d="${make_d} -DGPUJPEG_LIBRARIES=$lib_dir/gpujpeg_kar_gpu/lib/libgpujpeg.so"

#make_d="${make_d} -DGPUJPEG_INCLUDE_DIR="
#make_d="${make_d} -DGPUJPEG_LIBRARIES="


cmake ${make_d}
#make clean
make -j 2 install
#make install