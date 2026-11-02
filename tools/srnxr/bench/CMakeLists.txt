# SPDX-FileCopyrightText: 2011-2025 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

# Standalone SRNXR decoder tools, built separately from Cycles:
#   cmake -S tools/srnxr/bench -B build -DCMAKE_CUDA_ARCHITECTURES=120
#   cmake --build build
# SRNXR_OPTIX_ROOT pointing to an OptiX 9 SDK (include/optix.h) adds srnxr_coopvec.

cmake_minimum_required(VERSION 3.24)
project(srnxr_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(CYCLES_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../src" ABSOLUTE)

# Cycles compiles its CUDA kernels with these launch bounds (kernel/device/cuda/config.h,
# 384 threads, 168 registers on SM 7.x - 12.x), the _cycles_regs variant emulates them.
set(SRNXR_CYCLES_MAXRREG 168)

foreach(variant default cycles_regs)
  if(variant STREQUAL "default")
    set(target srnxr_bench)
  else()
    set(target srnxr_bench_${variant})
  endif()
  add_executable(${target} srnxr_bench.cu ${CYCLES_SRC}/scene/srnxr_model.cpp)
  target_include_directories(${target} PRIVATE ${CYCLES_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${target} PRIVATE SRNXR_STANDALONE=1)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v --expt-relaxed-constexpr>)
  if(variant STREQUAL "cycles_regs")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=${SRNXR_CYCLES_MAXRREG}>)
  endif()
  # srnxr_model.cpp is plain C++, srnxr_bench.cu includes the decoder for host and device.
  set_source_files_properties(${CYCLES_SRC}/scene/srnxr_model.cpp PROPERTIES LANGUAGE CXX)
  find_package(Threads REQUIRED)
  target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()

set(SRNXR_OPTIX_ROOT "" CACHE PATH "OptiX SDK 9.0+ root (include/optix.h) for srnxr_coopvec")
if(SRNXR_OPTIX_ROOT AND EXISTS "${SRNXR_OPTIX_ROOT}/include/optix.h")
  include(coopvec.cmake)
endif()
