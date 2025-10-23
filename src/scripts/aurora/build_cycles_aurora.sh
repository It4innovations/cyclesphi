#!/bin/bash

#qsub -I -l select=1 -l filesystems=flare -l walltime=1:00:00 -q debug -A Insitu
#ml oneapi/release/2024.2.1

cd ~/scratch/projects/cycles

ROOT_DIR=${PWD}

#ml oneapi/release/2024.2.1
#ml oneapi/eng-compiler/2024.07.30.002
#ml mpich
module restore
ml
ml cmake
##################################################

lib_dir=${ROOT_DIR}/install
output=${ROOT_DIR}/install/cycles_aurora_gpu
src=${ROOT_DIR}/src

export CC=gcc
export CXX=g++

#export CUDA_ROOT=/usr/local/cuda
#echo $CUDA_ROOT
#export OptiX_INSTALL_DIR=$lib_dir/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64
#export OptiX_INSTALL_DIR=$lib_dir/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64
#export LD_LIBRARY_PATH=${OptiX_INSTALL_DIR}:${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH
#export PATH=${CUDA_ROOT}/bin:$PATH

#export PATH=$PATH:/opt/aurora/24.347.0/oneapi/vtune/2025.0/bin64/gma/GTPin/Profilers/ocloc/Bin/intel64

# rm -rf ${ROOT_DIR}/build/cycles_aurora_gpu
# rm -rf ${ROOT_DIR}/install/cycles_aurora_gpu

#-----------blender_client--------------
mkdir ${ROOT_DIR}/build/cycles_aurora_gpu
cd ${ROOT_DIR}/build/cycles_aurora_gpu

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
make_d="${make_d} -DWITH_GPU_CPUIMAGE=ON"

#make_d="${make_d} -DOPTIX_ROOT_DIR=$OptiX_INSTALL_DIR"
#make_d="${make_d} -DCMAKE_CUDA_ARCHITECTURES=80"
make_d="${make_d} -DWITH_CYCLES_DEVICE_ONEAPI=ON"
make_d="${make_d} -DCYCLES_ONEAPI_INTEL_BINARIES_ARCH=pvc"
make_d="${make_d} -DWITH_CYCLES_ONEAPI_BINARIES=ON"
#CYCLES_ONEAPI_SYCL_TARGETS spir64
make_d="${make_d} -DCYCLES_ONEAPI_SYCL_TARGETS=spir64_gen" #spir64,spir64_gen" #spir64_gen"
#--intel -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc"
make_d="${make_d} -DCYCLES_ONEAPI_SYCL_OPTIONS_spir64_gen="
#make_d="${make_d} -DCYCLES_ONEAPI_INTEL_BINARIES_ARCH=pvc" # For Ponte Vecchio (Aurora)

make_d="${make_d} -DOCLOC_INSTALL_DIR=/usr/"

#make_d="${make_d} -DCYCLES_CUDA_BINARIES_ARCH=sm_80"
#make_d="${make_d} -DCUDA_BINARIES_ARCH:STRING=sm_80"
#make_d="${make_d} -DWITH_CUDA_DYNLOAD=OFF"

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

# sycl_compiler_flags: /home/jarosm/scratch/projects/cycles/src/cycles/src/kernel/device/oneapi/kernel.cpp;-fsycl;-fsycl-unnamed-lambda;-fdelayed-template-parsing;-fsycl-device-code-split=per_kernel;-fsycl-max-parallel-link-jobs=1;--offload-compress;--offload-compression-level=19;-shared;-DWITH_ONEAPI;-O2;-fno-fast-math;-ffp-contract=fast;-fassociative-math;-freciprocal-math;-fno-signed-zeros;-ffinite-math-only;-D__KERNEL_LOCAL_ATOMIC_SORT__;-o"/home/jarosm/scratch/projects/cycles/build/cycles_aurora_gpu/src/kernel/libcycles_kernel_oneapi_aot.so";-I"/home/jarosm/scratch/projects/cycles/src/cycles/src/kernel/..";--intel;-fsycl;-fsycl-targets=spir64_gen;-Xsycl-target-backend;"-device;pvc";-DWITH_NANOVDB;-DWITH_EMBREE;-DWITH_EMBREE_GPU;-DEMBREE_MAJOR_VERSION=4;-I"/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/include";$<$<CONFIG:Release>:/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4.so;/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4_sycl.a;/opt/aurora/24.347.0/oneapi/tbb/latest/lib/libtbb.so>;$<$<CONFIG:RelWithDebInfo>:/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4.so;/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4_sycl.a;/opt/aurora/24.347.0/oneapi/tbb/latest/lib/libtbb.so>;$<$<CONFIG:MinSizeRel>:/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4.so;/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4_sycl.a;/opt/aurora/24.347.0/oneapi/tbb/latest/lib/libtbb.so>;$<$<CONFIG:Debug>:/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4.so;/home/jarosm/scratch/projects/cycles/src/cycles/lib/linux_x64/embree/lib/libembree4_sycl.a;/opt/aurora/24.347.0/oneapi/tbb/latest/lib/libtbb.so>;-fPIC;-fvisibility=hidden;-Wl,-rpath,'$$ORIGIN'
