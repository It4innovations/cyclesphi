/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Minimal definitions that let the Cycles SRNXR decoder headers compile outside of Cycles,
 * for the host (reference) and the CUDA device (benchmarks). */

#define SRNXR_STANDALONE 1

#include <cfloat>
#include <cmath>
#include <cstdint>

#ifndef CCL_NAMESPACE_BEGIN
#  define CCL_NAMESPACE_BEGIN namespace ccl {
#  define CCL_NAMESPACE_END }
#endif

#ifdef __CUDACC__
#  define __KERNEL_CUDA__
#  ifdef __CUDA_ARCH__
#    define __KERNEL_GPU__
#  endif
#  define ccl_device __host__ __device__ inline
#  define ccl_device_inline __host__ __device__ inline
#  define ccl_device_forceinline __host__ __device__ __forceinline__
#  define ccl_device_noinline __host__ __device__ __noinline__
#else
#  define ccl_device inline
#  define ccl_device_inline inline
#  define ccl_device_forceinline inline
#  define ccl_device_noinline
#endif
#define ccl_global
#define ccl_private

CCL_NAMESPACE_BEGIN
#ifndef __CUDACC__
struct uint2 {
  uint32_t x, y;
};
struct uint4 {
  uint32_t x, y, z, w;
};
#endif
#ifdef __CUDACC__
__host__ __device__ inline uint32_t min(const uint32_t a, const uint32_t b)
{
  return a < b ? a : b;
}
#else
inline uint32_t min(const uint32_t a, const uint32_t b)
{
  return a < b ? a : b;
}
#endif
CCL_NAMESPACE_END

#include "kernel/util/image_3d_srnxr.h"
#include "kernel/util/image_3d_srnxr_decode.h"
