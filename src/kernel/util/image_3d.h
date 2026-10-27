/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/sample/lcg.h"

#include "util/types_image.h"
#include "kernel/util/image_3d_derivates.h"
#include "kernel/util/image_3d_srnxr.h"

#if !defined(__KERNEL_METAL__) && !defined(__KERNEL_ONEAPI__)
#  ifdef WITH_NANOVDB
#    include "kernel/util/nanovdb.h"
#    include <nanovdb/NanoVDB.h>
#  endif
#  ifdef WITH_ZFP_LOADER
#    include "kernel/util/zfp/array3_device.cuh"
#  endif
#endif

CCL_NAMESPACE_BEGIN

#ifndef __KERNEL_GPU__
/* Make template functions private so symbols don't conflict between kernels with different
 * instruction sets. */
namespace {
#endif

/* The volume formats of this file read NanoVDB grids with the NanoVDB library and compute
 * in double precision. Metal Shading Language has no doubles and cannot compile the NanoVDB
 * headers (no address space qualifiers): on Metal the grids are read with the extract in
 * `kernel/util/nanovdb.h` and the arithmetic is done in single precision. The code below is
 * written once, in terms of:
 *
 *   image_real_t                  type of the world and index space arithmetic
 *   IMAGE_NANOVDB                 namespace of the grid classes
 *   image_nanovdb_coord()         voxel coordinate
 *   image_nanovdb_world_to_index(), image_nanovdb_index_to_world(),
 *   image_nanovdb_voxel_size()    transform of a grid, the results are indexed with [] */
#ifdef __KERNEL_METAL__
typedef float image_real_t;
#  define IMAGE_NANOVDB ccl_nanovdb
#  define image_nanovdb_coord(ix, iy, iz) make_int3(ix, iy, iz)
#  define image_nanovdb_world_to_index(grid, x, y, z) \
    ((grid)->worldToIndexF(make_float3(x, y, z)))
#  define image_nanovdb_index_to_world(grid, x, y, z) \
    ((grid)->indexToWorldF(make_float3(x, y, z)))
#  define image_nanovdb_voxel_size(grid) ((grid)->voxelSizeF())
#else
typedef double image_real_t;
#  define IMAGE_NANOVDB nanovdb
#  define image_nanovdb_coord(ix, iy, iz) nanovdb::Coord(ix, iy, iz)
#  define image_nanovdb_world_to_index(grid, x, y, z) \
    ((grid)->worldToIndex(nanovdb::Vec3d(x, y, z)))
#  define image_nanovdb_index_to_world(grid, x, y, z) \
    ((grid)->indexToWorld(nanovdb::Vec3d(x, y, z)))
#  define image_nanovdb_voxel_size(grid) ((grid)->voxelSize())
#endif

/* CUB sparse voxel helper functions */
#define CUB_COORD_BITS 21
#define CUB_COORD_BIAS (1 << (CUB_COORD_BITS - 1))
#define CUB_COORD_MASK ((1ull << CUB_COORD_BITS) - 1ull)

ccl_device_inline uint64_t cub_pack_coord(int x, int y, int z)
{
  uint64_t ux = (uint64_t)(x + CUB_COORD_BIAS) & CUB_COORD_MASK;
  uint64_t uy = (uint64_t)(y + CUB_COORD_BIAS) & CUB_COORD_MASK;
  uint64_t uz = (uint64_t)(z + CUB_COORD_BIAS) & CUB_COORD_MASK;
  return (ux) | (uy << CUB_COORD_BITS) | (uz << (2 * CUB_COORD_BITS));
}

ccl_device_inline void cub_unpack_coord(uint64_t key,
                                        ccl_private int &x,
                                        ccl_private int &y,
                                        ccl_private int &z)
{
  uint64_t ux = (key) & CUB_COORD_MASK;
  uint64_t uy = (key >> CUB_COORD_BITS) & CUB_COORD_MASK;
  uint64_t uz = (key >> (2 * CUB_COORD_BITS)) & CUB_COORD_MASK;
  x = (int)ux - CUB_COORD_BIAS;
  y = (int)uy - CUB_COORD_BIAS;
  z = (int)uz - CUB_COORD_BIAS;
}

/* Binary search for a voxel in the sorted sparse array */
ccl_device_noinline int cub_find_voxel(const ccl_global uint64_t *keys,
                                       int count,
                                       uint64_t search_key)
{
  int left = 0;
  int right = count - 1;
  
  while (left <= right) {
    int mid = left + (right - left) / 2;
    uint64_t mid_key = keys[mid];
    
    if (mid_key == search_key) {
      return mid;
    }
    else if (mid_key < search_key) {
      left = mid + 1;
    }
    else {
      right = mid - 1;
    }
  }
  
  
  //for (int i= 0; i < count; i++) {
  //  if (keys[i] == search_key) {
  //    return i;
  //  }
  //}

  return -1; /* Not found */
}

/* Fetch voxel value from sparse array */
ccl_device_inline float cub_fetch(const ccl_global uint64_t *keys,
                                  const ccl_global float *values,
                                  int count,
                                  int x, int y, int z,
                                  int dimx, int dimy, int dimz)
{  
  /* Clamp coordinates to valid range */
  if (x < 0 || y < 0 || z < 0 || x >= dimx || y >= dimy || z >= dimz) {
    return 0.0f;
  }
  
  uint64_t search_key = cub_pack_coord(x, y, z);
  int index = cub_find_voxel(keys, count, search_key);
  
  if (index >= 0) {
    return values[index];
  }
  
  return 0.0f; /* Default value for empty voxels */
}

#ifdef WITH_NANOVDB

/* Cubic interpolation weights. */
ccl_device_forceinline void fill_cubic_weights(float3 w[4], float3 t)
{
  w[0] = (((-1.0f / 6.0f) * t + 0.5f) * t - 0.5f) * t + (1.0f / 6.0f);
  w[1] = ((0.5f * t - 1.0f) * t) * t + (2.0f / 3.0f);
  w[2] = ((-0.5f * t + 0.5f) * t + 0.5f) * t + (1.0f / 6.0f);
  w[3] = (1.0f / 6.0f) * t * t * t;
}

/* -------------------------------------------------------------------- */
/** Return the sample position for stochastical one-tap sampling.
 * From "Stochastic Texture Filtering": https://arxiv.org/abs/2305.05810
 * \{ */
ccl_device_inline float3 interp_tricubic_stochastic(const float3 P, ccl_private float3 &rand)
{
  const float3 p = floor(P);
  const float3 t = P - p;

  float3 w[4];
  fill_cubic_weights(w, t);

  /* For reservoir sampling, always accept the first in the stream. */
  float3 total_weight = w[0];
  float3 offset = make_float3(-1.0f);

  for (int j = 1; j < 4; j++) {
    total_weight += w[j];
    const float3 thresh = w[j] / total_weight;
    const auto mask = rand < thresh;
    offset = select(mask, make_float3(float(j) - 1.0f), offset);
    rand = select(mask, safe_divide(rand, thresh), safe_divide(rand - thresh, 1.0f - thresh));
  }

  return p + offset;
}

ccl_device_inline float3 interp_trilinear_stochastic(const float3 P, const float3 rand)
{
  const float3 p = floor(P);
  const float3 t = P - p;
  return select(rand < t, p + 1.0f, p);
}

ccl_device_inline float3 interp_stochastic(const float3 P,
                                           ccl_private InterpolationType &interpolation,
                                           ccl_private float3 &rand)
{
  float3 P_new = P;
  if (interpolation == INTERPOLATION_CUBIC) {
    P_new = interp_tricubic_stochastic(P, rand);
  }
  else if (interpolation == INTERPOLATION_LINEAR) {
    P_new = interp_trilinear_stochastic(P, rand);
  }
  else {
    kernel_assert(interpolation == INTERPOLATION_CLOSEST);
  }
  interpolation = INTERPOLATION_CLOSEST;
  return P_new;
}
/** \} */

template<typename OutT, typename Acc>
ccl_device OutT kernel_image_interp_trilinear_nanovdb(ccl_private Acc &acc, const float3 P)
{
  const float3 floor_P = floor(P);
  const float3 t = P - floor_P;
  const int3 index = make_int3(floor_P);

  const int ix = index.x;
  const int iy = index.y;
  const int iz = index.z;

  return mix(mix(mix(OutT(acc.getValue(make_int3(ix, iy, iz))),
                     OutT(acc.getValue(make_int3(ix, iy, iz + 1))),
                     t.z),
                 mix(OutT(acc.getValue(make_int3(ix, iy + 1, iz + 1))),
                     OutT(acc.getValue(make_int3(ix, iy + 1, iz))),
                     1.0f - t.z),
                 t.y),
             mix(mix(OutT(acc.getValue(make_int3(ix + 1, iy + 1, iz))),
                     OutT(acc.getValue(make_int3(ix + 1, iy + 1, iz + 1))),
                     t.z),
                 mix(OutT(acc.getValue(make_int3(ix + 1, iy, iz + 1))),
                     OutT(acc.getValue(make_int3(ix + 1, iy, iz))),
                     1.0f - t.z),
                 1.0f - t.y),
             t.x);
}

template<typename OutT, typename Acc>
ccl_device OutT kernel_image_interp_tricubic_nanovdb(ccl_private Acc &acc, const float3 P)
{
#  if defined(__KERNEL_HIP__)
  /* Explicitly unroll for HIP compiler to unroll the loop. Without this the render result is wrong
   * on a specific platform/compiler combinations. ALso don't rely on the `unroll` hint as it has
   * a performance impact. See #152126 and discussion/benchmark in !152321. */

  const float3 floor_P = floor(P);
  const float3 t = P - floor_P;
  const int3 index = make_int3(floor_P);

  const int xc[4] = {index.x - 1, index.x, index.x + 1, index.x + 2};
  const int yc[4] = {index.y - 1, index.y, index.y + 1, index.y + 2};
  const int zc[4] = {index.z - 1, index.z, index.z + 1, index.z + 2};

  float3 weight[4];
  fill_cubic_weights(weight, t);

#    define DATA(x, y, z) (OutT(acc.getValue(make_int3(xc[x], yc[y], zc[z]))))
#    define COL_TERM(col, row) \
      (weight[col].y * (weight[0].x * DATA(0, col, row) + weight[1].x * DATA(1, col, row) + \
                        weight[2].x * DATA(2, col, row) + weight[3].x * DATA(3, col, row)))
#    define ROW_TERM(row) \
      (weight[row].z * (COL_TERM(0, row) + COL_TERM(1, row) + COL_TERM(2, row) + COL_TERM(3, row)))

  /* Actual interpolation. */
  return ROW_TERM(0) + ROW_TERM(1) + ROW_TERM(2) + ROW_TERM(3);

#    undef COL_TERM
#    undef ROW_TERM
#    undef DATA

#  else

  const float3 floor_P = floor(P);
  const float3 t = P - floor_P;
  const int3 index = make_int3(floor_P) - make_int3(1);

  float3 w[4];
  fill_cubic_weights(w, t);

  OutT result = make_zero<OutT>();

  for (int k = 0; k < 4; k++) {
    OutT col_term_acc = make_zero<OutT>();
    for (int j = 0; j < 4; j++) {
      col_term_acc += w[j].y * (w[0].x * (OutT(acc.getValue(index + make_int3(0, j, k)))) +
                                w[1].x * (OutT(acc.getValue(index + make_int3(1, j, k)))) +
                                w[2].x * (OutT(acc.getValue(index + make_int3(2, j, k)))) +
                                w[3].x * (OutT(acc.getValue(index + make_int3(3, j, k)))));
    }
    result += w[k].z * col_term_acc;
  }

  return result;
#  endif
}

template<typename OutT, typename T>
#  if defined(__KERNEL_METAL__)
__attribute__((noinline))
#  else
ccl_device_noinline
#  endif
OutT kernel_image_interp_nanovdb(const ccl_global KernelImageInfo &info,
                                     float3 P,
                                     const InterpolationType interp)
{
#if 1

  ccl_global ccl_nanovdb::NanoGrid<T> *const grid = (ccl_global ccl_nanovdb::NanoGrid<T> *)
                                                        info.data;

  if (interp == INTERPOLATION_CLOSEST) {
    ccl_nanovdb::ReadAccessor<T> acc(grid->tree().root());
    return OutT(acc.getValue(make_int3(floor(P))));
  }

  ccl_nanovdb::CachedReadAccessor<T> acc(grid->tree().root());
  if (interp == INTERPOLATION_LINEAR) {
    return kernel_image_interp_trilinear_nanovdb<OutT>(acc, P);
  }

  return kernel_image_interp_tricubic_nanovdb<OutT>(acc, P);

#else

  ccl_global nanovdb::NanoGrid<T> *const grid = (ccl_global nanovdb::NanoGrid<T> *)info.data;
  // INTERPOLATION_CLOSEST
  nanovdb::ReadAccessor<T> acc(grid->tree().root());
  const nanovdb::Coord coord((int32_t)floorf(P.x), (int32_t)floorf(P.y), (int32_t)floorf(P.z));
  return OutT(acc.getValue(coord));

#endif
}

// ============================================================================
// Helpers for NanoVDB Derivative Bundle
// ============================================================================

// Taylor polynomial basis functions
// Maps derivative index to basis monomial for Taylor series reconstruction
// Given local offset (px, py, pz) from voxel center, returns the basis value
// Coefficients include factorial terms from Taylor expansion
ccl_device_inline image_real_t derivBasisValue(int basis,
                                               image_real_t p_x,
                                               image_real_t p_y,
                                               image_real_t p_z)
{
  switch (basis) {
    case 0:
      return 1.;

    // 1. derivation
    case 1:
      return p_x;
    case 2:
      return p_y;
    case 3:
      return p_z;

    // 2. derivation
    case 4:
      return 0.5 * (3 * p_x * p_x - 1);
    case 5:
      return 0.5 * (3 * p_y * p_y - 1);
    case 6:
      return 0.5 * (3 * p_z * p_z - 1);
    case 7:
      return p_x * p_y;
    case 8:
      return p_x * p_z;
    case 9:
      return p_y * p_z;

    // 3. derivation
    case 10:
      return 0.5 * (5 * p_x * p_x * p_x - 3 * p_x);
    case 11:
      return 0.5 * (5 * p_y * p_y * p_y - 3 * p_y);
    case 12:
      return 0.5 * (5 * p_z * p_z * p_z - 3 * p_z);
    case 13:
      return 0.5 * (3 * p_x * p_x - 1) * p_y;
    case 14:
      return 0.5 * (3 * p_x * p_x - 1) * p_z;
    case 15:
      return 0.5 * (3 * p_y * p_y - 1) * p_x;
    case 16:
      return 0.5 * (3 * p_y * p_y - 1) * p_z;
    case 17:
      return 0.5 * (3 * p_z * p_z - 1) * p_x;
    case 18:
      return 0.5 * (3 * p_z * p_z - 1) * p_y;
    case 19:
      return p_x * p_y * p_z;

    // 4. derivation
    case 20:
      return (35 * p_x * p_x * p_x * p_x - 30 * p_x * p_x + 3) / 8.0;
    case 21:
      return (35 * p_y * p_y * p_y * p_y - 30 * p_y * p_y + 3) / 8.0;
    case 22:
      return (35 * p_z * p_z * p_z * p_z - 30 * p_z * p_z + 3) / 8.0;
    case 23:
      return 0.5 * (5 * p_x * p_x * p_x - 3 * p_x) * p_y;
    case 24:
      return 0.5 * (5 * p_x * p_x * p_x - 3 * p_x) * p_z;
    case 25:
      return 0.5 * (5 * p_y * p_y * p_y - 3 * p_y) * p_x;
    case 26:
      return 0.5 * (5 * p_y * p_y * p_y - 3 * p_y) * p_z;
    case 27:
      return 0.5 * (5 * p_z * p_z * p_z - 3 * p_z) * p_x;
    case 28:
      return 0.5 * (5 * p_z * p_z * p_z - 3 * p_z) * p_y;
    case 29:
      return 0.5 * (3 * p_x * p_x - 1) * 0.5 * (3 * p_y * p_y - 1);
    case 30:
      return 0.5 * (3 * p_x * p_x - 1) * 0.5 * (3 * p_z * p_z - 1);
    case 31:
      return 0.5 * (3 * p_y * p_y - 1) * 0.5 * (3 * p_z * p_z - 1);
    case 32:
      return 0.5 * (3 * p_x * p_x - 1) * p_y * p_z;
    case 33:
      return 0.5 * (3 * p_y * p_y - 1) * p_x * p_z;
    case 34:
      return 0.5 * (3 * p_z * p_z - 1) * p_x * p_y;
  }
  return 0;
}

// Get pointer to NanoVDB grid from GridHeader
template<typename T>
ccl_device_inline const ccl_global IMAGE_NANOVDB::NanoGrid<T> *getDerivGridPtr(
    const ccl_global uint8_t *base, const ccl_global DerivGridHeader &gh)
{
  return reinterpret_cast<const ccl_global IMAGE_NANOVDB::NanoGrid<T> *>(base + gh.payloadOffset);
}

/* Whether a grid value is not zero. Metal Shading Language has no `if constexpr`, so the
 * value types are told apart by overloading. */
ccl_device_inline bool image_value_is_nonzero(const float f)
{
  return f != 0.0f;
}

ccl_device_inline bool image_value_is_nonzero(const float3 f)
{
  return f.x != 0.0f || f.y != 0.0f || f.z != 0.0f;
}

// ============================================================================
// Multi-Res (Old Format - Kept for Compatibility)
// ============================================================================

#if defined(__KERNEL_METAL__)
template<typename OutT, typename T>
__attribute__((noinline)) OutT kernel_tex_image_interp_nanovdb_multires(
    const ccl_global KernelImageInfo &info,
    const float x, const float y, const float z,
    const uint /*interpolation*/)
#else
template<typename OutT, typename T>
ccl_device_noinline OutT kernel_tex_image_interp_nanovdb_multires(
    const ccl_global KernelImageInfo &info,
    const float x, const float y, const float z,
    const uint /*interpolation*/)
#endif
{
    // Format description of bin file:
    // size_t : number of levels (aligned to 32 bytes)
    // size_t : offset to grid1
    // grid0 data (aligned to 32 bytes)
    // size_t : offset to grid2
    // grid1 data (aligned to 32 bytes)
    // ...

    size_t offset = 0;
    // Read number of levels
    const ccl_global char *data = (const ccl_global char *)info.data;
    size_t levels = *((const ccl_global size_t *)(data + offset));
    // Align to 32 bytes after num_levels
    offset = 32;

    for (size_t i = 0; i < levels; ++i) {
        // Get pointer to current grid data        
        const ccl_global char *grid_ptr = nullptr;
        size_t next_offset = 0;

        if (i < levels - 1) {
            // Read next grid offset
            next_offset = *((const ccl_global size_t *)(data + offset));
            // Grid data starts after the offset field
            grid_ptr = data + (offset + sizeof(size_t));
        } else {
            // Last grid has no offset field
            grid_ptr = data + offset;
        }

        const ccl_global IMAGE_NANOVDB::NanoGrid<T> *const grid =
            reinterpret_cast<const ccl_global IMAGE_NANOVDB::NanoGrid<T> *>(grid_ptr);

        const auto coord_index = image_nanovdb_world_to_index(
            grid, (image_real_t)x, (image_real_t)y, (image_real_t)z);

        IMAGE_NANOVDB::ReadAccessor<T> acc(grid->tree().root());
        const auto coord = image_nanovdb_coord((int32_t)floorf((float)coord_index[0]),
                                               (int32_t)floorf((float)coord_index[1]),
                                               (int32_t)floorf((float)coord_index[2]));
        OutT f = acc.getValue(coord);

        // For vector types, any component which is not zero counts.
        const bool is_nonzero = image_value_is_nonzero(f);

        if (is_nonzero) {
#  ifdef MULTIRES_COUNTER
#    ifdef __CUDA_ARCH__
          unsigned long long int *counter = const_cast<unsigned long long int *>(
              info_multires_level_counter + i);
          atomicAdd(counter, 1ULL);
#    endif
#  endif
            return f;
        }

        // advance
        if (i < levels - 1) {
            offset = next_offset - sizeof(size_t);
        }
    }

#  ifdef MULTIRES_COUNTER
#    ifdef __CUDA_ARCH__
    unsigned long long int *counter = const_cast<unsigned long long int *>(
        info_multires_level_counter + 15);
    atomicAdd(counter, 1ULL);
#    endif
#  endif

    return OutT(0.0f);
}

// ============================================================================
// Performance notes:
// - Read FileHeader once (64 bytes)
// - Read relevant LevelHeader (16 bytes)
// - Stream GridHeaders for the level (128 bytes each)
// - Each grid payload read uses NanoVDB ReadAccessor
// - Accumulate Taylor series coefficients * basis values
// ============================================================================

#if defined(__KERNEL_METAL__)
template<typename OutT, typename T>
__attribute__((noinline)) OutT kernel_tex_image_interp_nanovdb_derivates(
    const ccl_global KernelImageInfo &info,
    const float x, const float y, const float z,
    const uint /*interpolation*/)
#else
template<typename OutT, typename T>
ccl_device_noinline OutT kernel_tex_image_interp_nanovdb_derivates(
    const ccl_global KernelImageInfo &info,
    const float x, const float y, const float z,
    const uint /*interpolation*/)
#endif
{
    const ccl_global uint8_t *base = (const ccl_global uint8_t *)info.data;

    // Read FileHeader
    const ccl_global DerivFileHeader *fh = reinterpret_cast<const ccl_global DerivFileHeader *>(
        base);

    // Validate magic number (optional, can be removed for performance)
    // if (fh->magic != 0x4E56444D) return OutT(0.0f);

    const uint32_t levelCount = fh->levelCount;
    if (levelCount == 0) return OutT(0.0f);

    const ccl_global DerivLevelHeader *levelTable =
        reinterpret_cast<const ccl_global DerivLevelHeader *>(base + fh->levelTableOffset);
    const ccl_global DerivGridHeader *gridTable =
        reinterpret_cast<const ccl_global DerivGridHeader *>(base + fh->gridTableOffset);

    const image_real_t wx = (image_real_t)x, wy = (image_real_t)y, wz = (image_real_t)z;

    // Iterate through levels (coarsest to finest logic, or customize as needed)
    // Here we check each level for non-zero derivatives and reconstruct
    for (uint32_t levelIdx = 0; levelIdx < levelCount; ++levelIdx) {
        const ccl_global DerivLevelHeader &lh = levelTable[levelIdx];
        
        const uint32_t derivCount = lh.derivativeCount;
        const uint32_t firstGrid = lh.firstGridIndex;

        // Bounds check
        if (firstGrid + derivCount > fh->gridCount) continue;

        // Sample first grid to get reference point for local coordinates
        // All grids in a level share the same index space
        const ccl_global DerivGridHeader &gh0 = gridTable[firstGrid];
        const ccl_global IMAGE_NANOVDB::NanoGrid<T> *grid0 = getDerivGridPtr<T>(base, gh0);

        // Convert world to index space
        const auto ijk_d = image_nanovdb_world_to_index(grid0, wx, wy, wz);

        // Integer voxel coordinate
        const int32_t ix = (int32_t)floorf((float)ijk_d[0]);
        const int32_t iy = (int32_t)floorf((float)ijk_d[1]);
        const int32_t iz = (int32_t)floorf((float)ijk_d[2]);
        const auto coord = image_nanovdb_coord(ix, iy, iz);

        // Get current level's voxel size for proper normalization
        const auto current_voxel_size = image_nanovdb_voxel_size(grid0);
        const image_real_t current_voxel_size0 = current_voxel_size[0];
        const image_real_t current_voxel_size1 = current_voxel_size[1];
        const image_real_t current_voxel_size2 = current_voxel_size[2];

        // Local offset from voxel center for Taylor expansion
        // Calculate voxel center in index space and convert to world coordinates
        const image_real_t one_half = (image_real_t)0.5; /* `half` is a type in Metal. */
        const auto voxel_center_world = image_nanovdb_index_to_world(
            grid0,
            (image_real_t)ix + one_half,
            (image_real_t)iy + one_half,
            (image_real_t)iz + one_half);
        // Normalize by half the current level's voxel size to get [-1, 1] range
        // Following formula: p = (coo - center) / (gridSize / 2)
        const image_real_t px = (wx - voxel_center_world[0]) / (one_half * current_voxel_size0);
        const image_real_t py = (wy - voxel_center_world[1]) / (one_half * current_voxel_size1);
        const image_real_t pz = (wz - voxel_center_world[2]) / (one_half * current_voxel_size2);

        // Accumulate Taylor polynomial reconstruction
        image_real_t result = (image_real_t)0;
        bool hasNonZero = false;

        for (uint32_t d = 0; d < derivCount; ++d) {
            const ccl_global DerivGridHeader &gh = gridTable[firstGrid + d];
            const ccl_global IMAGE_NANOVDB::NanoGrid<T> *grid = getDerivGridPtr<T>(base, gh);

            // Read coefficient from grid
            IMAGE_NANOVDB::ReadAccessor<T> acc(grid->tree().root());
            const T coeff = acc.getValue(coord);

            // Check if coefficient is non-zero
            const bool nonzero = image_value_is_nonzero(coeff);

            if (nonzero) {
                hasNonZero = true;
                // Multiply coefficient by basis function and accumulate
                const image_real_t basis = derivBasisValue(gh.derivativeIndex, px, py, pz);
                result += (image_real_t)coeff * basis;
            }
        }

        // If this level has non-zero data, return the reconstruction
        if (hasNonZero) {
#ifdef MULTIRES_COUNTER
#ifdef __CUDA_ARCH__
            unsigned long long int *counter = const_cast<unsigned long long int*>(info_multires_level_counter + levelIdx);
            atomicAdd(counter, 1ULL);
#endif    
#endif
            return OutT(result);
        }
    }

    // No non-zero data found in any level
#ifdef MULTIRES_COUNTER
#ifdef __CUDA_ARCH__
    unsigned long long int *counter = const_cast<unsigned long long int*>(info_multires_level_counter + 15);
    atomicAdd(counter, 1ULL);
#endif    
#endif

    return OutT(0.0f);
}

// ============================================================================
// Vec4-Packed Derivative Format (New Format)
// ============================================================================
// This format packs up to 4 derivatives per grid to reduce grid count.
// Packed grid types: float (1), Vec3f (2-3), Vec4f (4)
// Raw derivative index = packedGridIndex * 4 + componentIndex
// ============================================================================

// Helper to read component from packed grid
#if defined(__KERNEL_METAL__)
/* No `if constexpr` and no NanoVDB vector types on Metal: one overload per grid type. */
ccl_device_inline float readPackedComponent(const float value, uint32_t componentIdx)
{
    return (componentIdx == 0) ? value : 0.0f;
}

ccl_device_inline float readPackedComponent(const packed_float3 value, uint32_t componentIdx)
{
    return (componentIdx == 0) ? value.x :
           (componentIdx == 1) ? value.y :
           (componentIdx == 2) ? value.z :
                                 0.0f;
}

ccl_device_inline float readPackedComponent(const float4 value, uint32_t componentIdx)
{
    return (componentIdx < 4) ? value[componentIdx] : 0.0f;
}
#else
template<typename T>
ccl_device_inline float readPackedComponent(const T& value, uint32_t componentIdx)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        // float grid - only component 0 is valid
        return (componentIdx == 0) ? value : 0.0f;
    }
    else if constexpr (sizeof(T) == sizeof(nanovdb::Vec3f)) {
        // Vec3f grid - components 0-2
        const auto& vec = reinterpret_cast<const nanovdb::Vec3f&>(value);
        return (componentIdx < 3) ? vec[componentIdx] : 0.0f;
    }
    else if constexpr (sizeof(T) == sizeof(nanovdb::Vec4f)) {
        // Vec4f grid - components 0-3
        const auto& vec = reinterpret_cast<const nanovdb::Vec4f&>(value);
        return (componentIdx < 4) ? vec[componentIdx] : 0.0f;
    }
    else {
        return 0.0f;
    }
}
#endif

#if defined(__KERNEL_METAL__)
template<typename OutT>
__attribute__((noinline)) OutT kernel_tex_image_interp_nanovdb_derivates_vec4(
    const ccl_global KernelImageInfo &info,
    const float x, const float y, const float z,
    const uint /*interpolation*/)
#else
template<typename OutT>
ccl_device_noinline OutT kernel_tex_image_interp_nanovdb_derivates_vec4(
    const ccl_global KernelImageInfo &info,
    const float x, const float y, const float z,
    const uint /*interpolation*/)
#endif
{
    const ccl_global uint8_t *base = (const ccl_global uint8_t *)info.data;

    // Read FileHeader
    const ccl_global DerivFileHeader *fh = reinterpret_cast<const ccl_global DerivFileHeader *>(
        base);

    const uint32_t levelCount = fh->levelCount;
    if (levelCount == 0) return OutT(0.0f);

    const ccl_global DerivLevelHeader *levelTable =
        reinterpret_cast<const ccl_global DerivLevelHeader *>(base + fh->levelTableOffset);
    const ccl_global DerivGridHeader *gridTable =
        reinterpret_cast<const ccl_global DerivGridHeader *>(base + fh->gridTableOffset);

    const image_real_t wx = (image_real_t)x, wy = (image_real_t)y, wz = (image_real_t)z;

    // Iterate through levels
    for (uint32_t levelIdx = 0; levelIdx < levelCount; ++levelIdx) {
        const ccl_global DerivLevelHeader &lh = levelTable[levelIdx];
        
        const uint32_t packedGridCount = lh.derivativeCount;  // Number of packed grids
        const uint32_t firstGrid = lh.firstGridIndex;

        // Bounds check
        if (firstGrid + packedGridCount > fh->gridCount) continue;

        // Sample first grid to get reference point for local coordinates
        const ccl_global DerivGridHeader &gh0 = gridTable[firstGrid];
        const ccl_global IMAGE_NANOVDB::NanoGrid<float> *grid0 =
            reinterpret_cast<const ccl_global IMAGE_NANOVDB::NanoGrid<float> *>(
                base + gh0.payloadOffset);

        // Convert world to index space
        const auto ijk_d = image_nanovdb_world_to_index(grid0, wx, wy, wz);

        // Integer voxel coordinate
        const int32_t ix = (int32_t)floorf((float)ijk_d[0]);
        const int32_t iy = (int32_t)floorf((float)ijk_d[1]);
        const int32_t iz = (int32_t)floorf((float)ijk_d[2]);
        const auto coord = image_nanovdb_coord(ix, iy, iz);

        // Get current level's voxel size for proper normalization
        const auto current_voxel_size = image_nanovdb_voxel_size(grid0);
        const image_real_t current_voxel_size0 = current_voxel_size[0];
        const image_real_t current_voxel_size1 = current_voxel_size[1];
        const image_real_t current_voxel_size2 = current_voxel_size[2];

        // Local offset from voxel center for Taylor expansion
        const image_real_t one_half = (image_real_t)0.5; /* `half` is a type in Metal. */
        const auto voxel_center_world = image_nanovdb_index_to_world(
            grid0,
            (image_real_t)ix + one_half,
            (image_real_t)iy + one_half,
            (image_real_t)iz + one_half);
        // Normalize by half the current level's voxel size to get [-1, 1] range
        // Following formula: p = (coo - center) / (gridSize / 2)
        const image_real_t px = (wx - voxel_center_world[0]) / (one_half * current_voxel_size0);
        const image_real_t py = (wy - voxel_center_world[1]) / (one_half * current_voxel_size1);
        const image_real_t pz = (wz - voxel_center_world[2]) / (one_half * current_voxel_size2);

        // Accumulate Taylor polynomial reconstruction
        image_real_t result = (image_real_t)0;
        bool hasNonZero = false;

        // Iterate through packed grids in this level
        for (uint32_t packedIdx = 0; packedIdx < packedGridCount; ++packedIdx) {
            const ccl_global DerivGridHeader &gh = gridTable[firstGrid + packedIdx];
            
            // Base derivative index for this packed grid
            const uint32_t baseDerivIdx = packedIdx * 4;

            // Dispatch based on grid type
            const DerivGridType gridType = (DerivGridType)gh.gridType;
            
            if (gridType == DERIV_GRID_TYPE_FLOAT) {
                // Float grid - 1 component
                const ccl_global IMAGE_NANOVDB::NanoGrid<float> *grid =
                    reinterpret_cast<const ccl_global IMAGE_NANOVDB::NanoGrid<float> *>(
                        base + gh.payloadOffset);
                IMAGE_NANOVDB::ReadAccessor<float> acc(grid->tree().root());
                const float coeff = acc.getValue(coord);

                if (coeff != 0.0f) {
                    hasNonZero = true;
                    const image_real_t basis = derivBasisValue(baseDerivIdx, px, py, pz);
                    result += (image_real_t)coeff * basis;
                }
            }
            else if (gridType == DERIV_GRID_TYPE_VEC3F) {
                // Vec3f grid - up to 3 components
                const ccl_global IMAGE_NANOVDB::NanoGrid<packed_float3> *grid =
                    reinterpret_cast<const ccl_global IMAGE_NANOVDB::NanoGrid<packed_float3> *>(
                        base + gh.payloadOffset);
                IMAGE_NANOVDB::ReadAccessor<packed_float3> acc(grid->tree().root());
                const packed_float3 packedValue = acc.getValue(coord);

                for (uint32_t compIdx = 0; compIdx < 3; ++compIdx) {
                    const float coeff = readPackedComponent(packedValue, compIdx);
                    if (coeff != 0.0f) {
                        hasNonZero = true;
                        const uint32_t derivIdx = baseDerivIdx + compIdx;
                        const image_real_t basis = derivBasisValue(derivIdx, px, py, pz);
                        result += (image_real_t)coeff * basis;
                    }
                }
            }
            else if (gridType == DERIV_GRID_TYPE_VEC4F) {
                // Vec4f grid - 4 components
                const ccl_global IMAGE_NANOVDB::NanoGrid<float4> *grid =
                    reinterpret_cast<const ccl_global IMAGE_NANOVDB::NanoGrid<float4> *>(
                        base + gh.payloadOffset);
                IMAGE_NANOVDB::ReadAccessor<float4> acc(grid->tree().root());
                const float4 packedValue = acc.getValue(coord);

                for (uint32_t compIdx = 0; compIdx < 4; ++compIdx) {
                    const float coeff = readPackedComponent(packedValue, compIdx);
                    if (coeff != 0.0f) {
                        hasNonZero = true;
                        const uint32_t derivIdx = baseDerivIdx + compIdx;
                        const image_real_t basis = derivBasisValue(derivIdx, px, py, pz);
                        result += (image_real_t)coeff * basis;
                    }
                }
            }
        }

        // If this level has non-zero data, return the reconstruction
        if (hasNonZero) {
#ifdef MULTIRES_COUNTER
#ifdef __CUDA_ARCH__
            unsigned long long int *counter = const_cast<unsigned long long int*>(
                info_multires_level_counter + levelIdx);
            atomicAdd(counter, 1ULL);
#endif    
#endif
            return OutT(result);
        }
    }

    // No non-zero data found in any level
#ifdef MULTIRES_COUNTER
#ifdef __CUDA_ARCH__
    unsigned long long int *counter = const_cast<unsigned long long int*>(
        info_multires_level_counter + 15);
    atomicAdd(counter, 1ULL);
#endif    
#endif

    return OutT(0.0f);
}

#endif /* WITH_NANOVDB */

// ============================================================================
// SRNXR (SIREN octree) Format
// ============================================================================
// The density is evaluated at P directly from the network weights in device
// memory, there is no voxel data:
//   q  = shift(P)                          off-grid modulo shift of the domain
//   xi = (q - center) / scale              for every candidate block of P's grid
//                                          cell, the block contains P when
//                                          max|xi| <= overlap
//   y  = MLP(xi / overlap) * sigma + mu    sine layers, linear head, block denorm
//   w  = exp(-0.5 |xi|^2 / blend_sigma^2)  blending weight
//   density = clamp(sum(w y) / sum(w) * value_std + value_mean, 0, 1)
// The density is 0 where sum(w) is below weight_sum_epsilon or P is outside
// the domain.
// ============================================================================

ccl_device_forceinline float srnxr_sin(const float x)
{
#if defined(__KERNEL_CUDA__) || defined(__KERNEL_HIP__)
  /* Hardware intrinsic, its error is far below the FP16 quantization of the weights. */
  return __sinf(x);
#else
  return sinf(x);
#endif
}

ccl_device_forceinline float srnxr_mod1(const float value)
{
  return value - floorf(value);
}

/* Network of one block: sine [width, 3], (hidden_layer_count - 1) x sine [width, width],
 * linear head [1, width]. */
ccl_device_forceinline float srnxr_eval_block(const ccl_global float *weights,
                                              const ccl_global float *biases,
                                              const uint32_t width,
                                              const uint32_t hidden_layer_count,
                                              const float w0,
                                              const float in0,
                                              const float in1,
                                              const float in2)
{
  float activation[SRNXR_MAX_WIDTH];
  float next[SRNXR_MAX_WIDTH];

  for (uint32_t o = 0; o < width; ++o) {
    const ccl_global float *row = weights + o * 3;
    activation[o] = srnxr_sin(w0 * (row[0] * in0 + row[1] * in1 + row[2] * in2 + biases[o]));
  }
  weights += width * 3;
  biases += width;

  for (uint32_t layer = 1; layer < hidden_layer_count; ++layer) {
    for (uint32_t o = 0; o < width; ++o) {
      const ccl_global float *row = weights + o * width;
      float sum = biases[o];
      for (uint32_t i = 0; i < width; ++i) {
        sum += row[i] * activation[i];
      }
      next[o] = srnxr_sin(w0 * sum);
    }
    for (uint32_t o = 0; o < width; ++o) {
      activation[o] = next[o];
    }
    weights += width * width;
    biases += width;
  }

  float value = biases[0];
  for (uint32_t i = 0; i < width; ++i) {
    value += weights[i] * activation[i];
  }
  return value;
}

ccl_device_noinline float kernel_image_interp_srnxr(const ccl_global KernelImageInfo &info,
                                                    const float3 P)
{
  const ccl_global uint8_t *base = (const ccl_global uint8_t *)info.data;
  const ccl_global SRNXRDeviceHeader *hdr =
      reinterpret_cast<const ccl_global SRNXRDeviceHeader *>(base);

  const float p[3] = {P.x, P.y, P.z};
  const float domain_min[3] = {hdr->domain_min_x, hdr->domain_min_y, hdr->domain_min_z};
  const float domain_max[3] = {hdr->domain_max_x, hdr->domain_max_y, hdr->domain_max_z};
  const float shift[3] = {hdr->shift_x, hdr->shift_y, hdr->shift_z};
  const int grid_dim[3] = {(int)hdr->grid_dim_x, (int)hdr->grid_dim_y, (int)hdr->grid_dim_z};

  // Shifted position (blocks live in shifted coordinates) and world-space grid cell
  float shifted[3];
  int cell[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (!(p[axis] >= domain_min[axis] && p[axis] <= domain_max[axis])) {
      return 0.0f;
    }
    const float size = domain_max[axis] - domain_min[axis];
    const float normalized = (p[axis] - domain_min[axis]) / size;
    shifted[axis] = srnxr_mod1(normalized + shift[axis]) * size + domain_min[axis];

    const float cell_size = size / (float)grid_dim[axis];
    cell[axis] = clamp((int)floorf((p[axis] - domain_min[axis]) / cell_size), 0, grid_dim[axis] - 1);
  }

  const ccl_global float *block_centers = reinterpret_cast<const ccl_global float *>(
      base + hdr->block_centers_offset);
  const ccl_global float *block_scales = reinterpret_cast<const ccl_global float *>(
      base + hdr->block_scales_offset);
  const ccl_global float *block_stats = reinterpret_cast<const ccl_global float *>(
      base + hdr->block_stats_offset);
  const ccl_global uint32_t *block_layout = reinterpret_cast<const ccl_global uint32_t *>(
      base + hdr->block_layout_offset);
  const ccl_global float *weights = reinterpret_cast<const ccl_global float *>(
      base + hdr->siren_weights_offset);
  const ccl_global float *biases = reinterpret_cast<const ccl_global float *>(
      base + hdr->siren_biases_offset);
  const ccl_global uint32_t *grid_offsets = reinterpret_cast<const ccl_global uint32_t *>(
      base + hdr->grid_offsets_offset);
  const ccl_global uint32_t *grid_indices = reinterpret_cast<const ccl_global uint32_t *>(
      base + hdr->grid_indices_offset);

  // Grid cells are stored x-major
  const uint32_t cell_index = ((uint32_t)cell[0] * (uint32_t)grid_dim[1] + (uint32_t)cell[1]) *
                                  (uint32_t)grid_dim[2] +
                              (uint32_t)cell[2];
  const uint32_t candidate_begin = grid_offsets[cell_index];
  const uint32_t candidate_end = grid_offsets[cell_index + 1];

  const float w0 = hdr->w0;
  const float overlap = hdr->overlap;
  const uint32_t hidden_layer_count = hdr->hidden_layer_count;
  const float inv_two_sigma_sq = 0.5f / (hdr->blend_sigma * hdr->blend_sigma);

  float numerator = 0.0f;
  float weight_sum = 0.0f;

  for (uint32_t candidate = candidate_begin; candidate < candidate_end; ++candidate) {
    const uint32_t block = grid_indices[candidate];
    const ccl_global float *center = block_centers + block * 3;
    const ccl_global float *scale = block_scales + block * 3;
    const float xi0 = (shifted[0] - center[0]) / scale[0];
    const float xi1 = (shifted[1] - center[1]) / scale[1];
    const float xi2 = (shifted[2] - center[2]) / scale[2];
    if (fabsf(xi0) > overlap || fabsf(xi1) > overlap || fabsf(xi2) > overlap) {
      continue;
    }

    const ccl_global uint32_t *layout = block_layout + block * SRNXR_BLOCK_LAYOUT_FIELDS;
    const uint32_t width = layout[0];
    const ccl_global float *block_weights = weights + layout[2];
    const ccl_global float *block_biases = biases + layout[3];

    const float in0 = xi0 / overlap;
    const float in1 = xi1 / overlap;
    const float in2 = xi2 / overlap;

    // Block network
    float value = srnxr_eval_block(
        block_weights, block_biases, width, hidden_layer_count, w0, in0, in1, in2);

    // Per-block denormalization
    const ccl_global float *stats = block_stats + block * SRNXR_BLOCK_STATS_FIELDS;
    value = value * stats[1] + stats[0];

    const float distance_sq = xi0 * xi0 + xi1 * xi1 + xi2 * xi2;
    const float blend_weight = expf(-distance_sq * inv_two_sigma_sq);
    numerator += blend_weight * value;
    weight_sum += blend_weight;
  }

  if (weight_sum <= hdr->weight_sum_epsilon) {
    return 0.0f;
  }
  const float value = (numerator / weight_sum) * hdr->value_std + hdr->value_mean;
  return clamp(value, 0.0f, 1.0f);
}

// ============================================================================
// Helpers for RAW3D / ZFP interpolation
// ============================================================================

#ifdef WITH_NANOVDB
ccl_device_forceinline float raw3d_fetch(const ccl_global float *data,
                                          size_t dimx,
                                          size_t dimy,
                                          size_t dimz,
                                          int ix,
                                          int iy,
                                          int iz)
{
  if (ix < 0 || iy < 0 || iz < 0 || (size_t)ix >= dimx || (size_t)iy >= dimy ||
      (size_t)iz >= dimz)
    return 0.0f;
  return data[(size_t)ix + (size_t)iy * dimx + (size_t)iz * dimx * dimy];
}

#  ifdef WITH_ZFP_LOADER
ccl_device_forceinline float zfp_fetch(cuZFP::DeviceBlockStore3<float, 64> &store,
                                        size_t dimx,
                                        size_t dimy,
                                        size_t dimz,
                                        int ix,
                                        int iy,
                                        int iz)
{
  if (ix < 0 || iy < 0 || iz < 0 || (size_t)ix >= dimx || (size_t)iy >= dimy ||
      (size_t)iz >= dimz)
    return 0.0f;
  const size_t block_idx = store.get_block_index((size_t)ix, (size_t)iy, (size_t)iz);
  const uint local_idx = ((uint)ix & 3u) + 4u * (((uint)iy & 3u) + 4u * ((uint)iz & 3u));
  cuZFP::NoCache<float, 64> cache;
  return cache.get(store, block_idx, local_idx);
}
#  endif /* WITH_ZFP_LOADER */
#endif /* WITH_NANOVDB */

// ============================================================================
// ZFP Compressed Format Support
// ============================================================================
ccl_device float4 kernel_image_interp_3d(KernelGlobals kg,
                                         ccl_private ShaderData *sd,
                                         const int image_texture_id,
                                         float3 P,
                                         InterpolationType interp,
                                         const bool stochastic)
{
#ifdef WITH_NANOVDB
  const ccl_global KernelImageTexture &tex = kernel_data_fetch(image_textures, image_texture_id);
  const ccl_global KernelImageInfo &info = kernel_data_fetch(image_info, tex.image_info_id);

  if (tex.use_transform_3d) {
    P = transform_point(&tex.transform_3d, P);
  }

  /* SRNXR is a continuous field evaluated at P in object space, it needs neither voxel
   * interpolation nor the stochastic jitter. */
  if ((ImageDataType)info.data_type == IMAGE_DATA_TYPE_SRNXR_FLOAT) {
    const float f = kernel_image_interp_srnxr(info, P);
    return make_float4(f, f, f, 1.0f);
  }

  InterpolationType interpolation = (interp == INTERPOLATION_NONE) ?
                                        (InterpolationType)info.interpolation :
                                        interp;

  /* RAW3D and ZFP grids pack their object to voxel index mapping into transform_3d (dimensions,
   * scale, translation and bounding box minimum in the columns) rather than storing a matrix. Map
   * P to voxel index space before the stochastic jitter, which works in index space. */
  const ImageDataType packed_type = (ImageDataType)info.data_type;
  if (packed_type == IMAGE_DATA_TYPE_RAW3D_FLOAT || packed_type == IMAGE_DATA_TYPE_ZFP_FLOAT) {
    P = make_float3((P.x - tex.transform_3d.x.w - tex.transform_3d.x.z) / tex.transform_3d.x.y,
                    (P.y - tex.transform_3d.y.w - tex.transform_3d.y.z) / tex.transform_3d.y.y,
                    (P.z - tex.transform_3d.z.w - tex.transform_3d.z.z) / tex.transform_3d.z.y);
  }

  if (stochastic) {
    float3 rand = lcg_step_float3(&sd->lcg_state);
    P = interp_stochastic(P, interpolation, rand);
  }

  const ImageDataType data_type = (ImageDataType)info.data_type;
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FLOAT) {
    const float f = kernel_image_interp_nanovdb<float, float>(info, P, interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FLOAT3) {
    const float3 f = kernel_image_interp_nanovdb<float3, packed_float3>(info, P, interpolation);
    return make_float4(f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FLOAT4) {
    return kernel_image_interp_nanovdb<float4, float4>(info, P, interpolation);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FPN) {
    const float f = kernel_image_interp_nanovdb<float, ccl_nanovdb::FpN>(info, P, interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FP16) {
    const float f = kernel_image_interp_nanovdb<float, ccl_nanovdb::Fp16>(info, P, interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT) {
    const float f = kernel_tex_image_interp_nanovdb_multires<float, float>(info, P.x, P.y, P.z, (uint)interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_DERIVATES) {
    const float f = kernel_tex_image_interp_nanovdb_derivates<float, float>(info, P.x, P.y, P.z, (uint)interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_DERIVATES_VEC4) {
    const float f = kernel_tex_image_interp_nanovdb_derivates_vec4<float>(info, P.x, P.y, P.z, (uint)interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_EMPTY) {
    return zero_float4();
  }
  if (data_type == IMAGE_DATA_TYPE_RAW3D_FLOAT) {

    const size_t dimx = (size_t)tex.transform_3d.x.x;
    const size_t dimy = (size_t)tex.transform_3d.y.x;
    const size_t dimz = (size_t)tex.transform_3d.z.x;

    /* P is in voxel index space already. */
    float px = P.x;
    float py = P.y;
    float pz = P.z;

    if (px < 0.0f || py < 0.0f || pz < 0.0f)
      return zero_float4();

    if (floorf(px) >= dimx || 
        floorf(py) >= dimy || 
        floorf(pz) >= dimz) {
      return zero_float4();
    }

    const ccl_global float *data = (const ccl_global float *)info.data;

    if (interpolation == INTERPOLATION_LINEAR) {
      const int ix0 = (int)floorf(px), iy0 = (int)floorf(py), iz0 = (int)floorf(pz);
      const float tx = px - (float)ix0, ty = py - (float)iy0, tz = pz - (float)iz0;
      const int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
      const float f = mix(
          mix(mix(raw3d_fetch(data, dimx, dimy, dimz, ix0, iy0, iz0),
                  raw3d_fetch(data, dimx, dimy, dimz, ix0, iy0, iz1),
                  tz),
              mix(raw3d_fetch(data, dimx, dimy, dimz, ix0, iy1, iz1),
                  raw3d_fetch(data, dimx, dimy, dimz, ix0, iy1, iz0),
                  1.0f - tz),
              ty),
          mix(mix(raw3d_fetch(data, dimx, dimy, dimz, ix1, iy1, iz0),
                  raw3d_fetch(data, dimx, dimy, dimz, ix1, iy1, iz1),
                  tz),
              mix(raw3d_fetch(data, dimx, dimy, dimz, ix1, iy0, iz1),
                  raw3d_fetch(data, dimx, dimy, dimz, ix1, iy0, iz0),
                  1.0f - tz),
              1.0f - ty),
          tx);
      return make_float4(f, f, f, 1.0f);
    }

    if (interpolation == INTERPOLATION_CUBIC) {
      const float3 fp = make_float3(floorf(px), floorf(py), floorf(pz));
      const float3 t = make_float3(px, py, pz) - fp;
      const int3 idx = make_int3((int)fp.x - 1, (int)fp.y - 1, (int)fp.z - 1);

      float3 w[4];
      fill_cubic_weights(w, t);

      float result = 0.0f;
      for (int k = 0; k < 4; k++) {
        float col_acc = 0.0f;
        for (int j = 0; j < 4; j++) {
          col_acc +=
              w[j].y *
              (w[0].x * raw3d_fetch(data, dimx, dimy, dimz, idx.x + 0, idx.y + j, idx.z + k) +
               w[1].x * raw3d_fetch(data, dimx, dimy, dimz, idx.x + 1, idx.y + j, idx.z + k) +
               w[2].x * raw3d_fetch(data, dimx, dimy, dimz, idx.x + 2, idx.y + j, idx.z + k) +
               w[3].x * raw3d_fetch(data, dimx, dimy, dimz, idx.x + 3, idx.y + j, idx.z + k));
        }
        result += w[k].z * col_acc;
      }
      return make_float4(result, result, result, 1.0f);
    }

    /* INTERPOLATION_CLOSEST */
    const size_t ix = (size_t)(floorf(px));
    const size_t iy = (size_t)(floorf(py));
    const size_t iz = (size_t)(floorf(pz));
    const size_t index = ix + iy * dimx + iz * dimx * dimy;
    const float f = data[index];
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_RAW3D_FLOAT3) {
    //TODO
    return zero_float4();
  }
#ifdef WITH_ZFP_LOADER
  if (data_type == IMAGE_DATA_TYPE_ZFP_FLOAT) {
    const size_t dimx = (size_t)tex.transform_3d.x.x;
    const size_t dimy = (size_t)tex.transform_3d.y.x;
    const size_t dimz = (size_t)tex.transform_3d.z.x;

    /* P is in voxel index space already. */
    float px = P.x;
    float py = P.y;
    float pz = P.z;

    if (px < 0.0f || py < 0.0f || pz < 0.0f)
      return zero_float4();

    if (floorf(px) >= dimx || floorf(py) >= dimy || floorf(pz) >= dimz) {
      return zero_float4();
    }

    const SerializableZFPData* header = (const SerializableZFPData*)info.data;

    typedef unsigned long long Word;
    Word* compressed_data_ptr = (Word*)((char*)info.data + header->placeholder_ptr);

    cuZFP::DeviceBlockStore3<float, 64> store;
    store.d_compressed_data = compressed_data_ptr;
    store.d_block_offsets = nullptr;
    store.dims = make_uint3(header->dims_x, header->dims_y, header->dims_z);
    store.block_dims = make_uint3(header->block_dims_x, header->block_dims_y, header->block_dims_z);
    store.maxbits = header->maxbits;
    store.total_blocks = header->total_blocks;
    store.fixed_rate = (header->fixed_rate != 0);

    if (interpolation == INTERPOLATION_LINEAR) {
      const int ix0 = (int)floorf(px), iy0 = (int)floorf(py), iz0 = (int)floorf(pz);
      const float tx = px - (float)ix0, ty = py - (float)iy0, tz = pz - (float)iz0;
      const int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
      const float f = mix(
          mix(mix(zfp_fetch(store, dimx, dimy, dimz, ix0, iy0, iz0),
                  zfp_fetch(store, dimx, dimy, dimz, ix0, iy0, iz1),
                  tz),
              mix(zfp_fetch(store, dimx, dimy, dimz, ix0, iy1, iz1),
                  zfp_fetch(store, dimx, dimy, dimz, ix0, iy1, iz0),
                  1.0f - tz),
              ty),
          mix(mix(zfp_fetch(store, dimx, dimy, dimz, ix1, iy1, iz0),
                  zfp_fetch(store, dimx, dimy, dimz, ix1, iy1, iz1),
                  tz),
              mix(zfp_fetch(store, dimx, dimy, dimz, ix1, iy0, iz1),
                  zfp_fetch(store, dimx, dimy, dimz, ix1, iy0, iz0),
                  1.0f - tz),
              1.0f - ty),
          tx);
      return make_float4(f, f, f, 1.0f);
    }

    if (interpolation == INTERPOLATION_CUBIC) {
      const float3 fp = make_float3(floorf(px), floorf(py), floorf(pz));
      const float3 t = make_float3(px, py, pz) - fp;
      const int3 idx = make_int3((int)fp.x - 1, (int)fp.y - 1, (int)fp.z - 1);

      float3 w[4];
      fill_cubic_weights(w, t);

      float result = 0.0f;
      for (int k = 0; k < 4; k++) {
        float col_acc = 0.0f;
        for (int j = 0; j < 4; j++) {
          col_acc +=
              w[j].y *
              (w[0].x * zfp_fetch(store, dimx, dimy, dimz, idx.x + 0, idx.y + j, idx.z + k) +
               w[1].x * zfp_fetch(store, dimx, dimy, dimz, idx.x + 1, idx.y + j, idx.z + k) +
               w[2].x * zfp_fetch(store, dimx, dimy, dimz, idx.x + 2, idx.y + j, idx.z + k) +
               w[3].x * zfp_fetch(store, dimx, dimy, dimz, idx.x + 3, idx.y + j, idx.z + k));
        }
        result += w[k].z * col_acc;
      }
      return make_float4(result, result, result, 1.0f);
    }

    /* INTERPOLATION_CLOSEST */
    const size_t ix = (size_t)(floorf(px));
    const size_t iy = (size_t)(floorf(py));
    const size_t iz = (size_t)(floorf(pz));
    const size_t block_idx = store.get_block_index(ix, iy, iz);
    const uint local_idx = (ix & 3) + 4 * ((iy & 3) + 4 * (iz & 3));
    cuZFP::NoCache<float, 64> cache;
    const float f = cache.get(store, block_idx, local_idx);
    return make_float4(f, f, f, 1.0f);
  }
#endif
  if (data_type == IMAGE_DATA_TYPE_CUB_FLOAT) {
    // CUB now uses transform from header similar to NanoVDB
    const ccl_global SerializableCUBData *header = (const ccl_global SerializableCUBData *)
                                                       info.data;
    size_t header_size = sizeof(SerializableCUBData);
    const int voxel_count = header->voxel_count;
    
    // Extract bbox from header
    const float bbox_min_x = header->bbox[0];
    const float bbox_min_y = header->bbox[1];
    const float bbox_min_z = header->bbox[2];
    const float bbox_max_x = header->bbox[3];
    const float bbox_max_y = header->bbox[4];
    const float bbox_max_z = header->bbox[5];
    
    const int dimx = (int)(bbox_max_x - bbox_min_x) + 1;
    const int dimy = (int)(bbox_max_y - bbox_min_y) + 1;
    const int dimz = (int)(bbox_max_z - bbox_min_z) + 1;
    
    // Apply inverse transform to get from world space to index space
    // transform_3d already contains the inverse in metadata
    //float3 P_transformed = transform_point(&tex.transform_3d, P);
    
    float px = P.x;
    float py = P.y;
    float pz = P.z;

    if (px < 0.0f || py < 0.0f || pz < 0.0f)
      return zero_float4();

    if (floorf(px) >= dimx || floorf(py) >= dimy || floorf(pz) >= dimz) {
      return zero_float4();
    }
    
    const ccl_global char *cub_data = (const ccl_global char *)info.data;
    const ccl_global uint64_t *keys = (const ccl_global uint64_t *)(cub_data + header_size);
    const ccl_global float *values = (const ccl_global float *)(cub_data + header_size +
                                                                voxel_count * sizeof(uint64_t));

    if (interpolation == INTERPOLATION_LINEAR) {
      const int ix0 = (int)floorf(px), iy0 = (int)floorf(py), iz0 = (int)floorf(pz);
      const float tx = px - (float)ix0, ty = py - (float)iy0, tz = pz - (float)iz0;
      const int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;
      const float f = mix(
          mix(mix(cub_fetch(keys, values, voxel_count, ix0, iy0, iz0, dimx, dimy, dimz),
                  cub_fetch(keys, values, voxel_count, ix0, iy0, iz1, dimx, dimy, dimz),
                  tz),
              mix(cub_fetch(keys, values, voxel_count, ix0, iy1, iz1, dimx, dimy, dimz),
                  cub_fetch(keys, values, voxel_count, ix0, iy1, iz0, dimx, dimy, dimz),
                  1.0f - tz),
              ty),
          mix(mix(cub_fetch(keys, values, voxel_count, ix1, iy1, iz0, dimx, dimy, dimz),
                  cub_fetch(keys, values, voxel_count, ix1, iy1, iz1, dimx, dimy, dimz),
                  tz),
              mix(cub_fetch(keys, values, voxel_count, ix1, iy0, iz1, dimx, dimy, dimz),
                  cub_fetch(keys, values, voxel_count, ix1, iy0, iz0, dimx, dimy, dimz),
                  1.0f - tz),
              1.0f - ty),
          tx);
      return make_float4(f, f, f, 1.0f);
    }

    if (interpolation == INTERPOLATION_CUBIC) {
      const float3 fp = make_float3(floorf(px), floorf(py), floorf(pz));
      const float3 t = make_float3(px, py, pz) - fp;
      const int3 idx = make_int3((int)fp.x - 1, (int)fp.y - 1, (int)fp.z - 1);

      float3 w[4];
      fill_cubic_weights(w, t);

      float result = 0.0f;
      for (int k = 0; k < 4; k++) {
        float col_acc = 0.0f;
        for (int j = 0; j < 4; j++) {
          col_acc +=
              w[j].y *
              (w[0].x * cub_fetch(keys, values, voxel_count, idx.x + 0, idx.y + j, idx.z + k, dimx, dimy, dimz) +
               w[1].x * cub_fetch(keys, values, voxel_count, idx.x + 1, idx.y + j, idx.z + k, dimx, dimy, dimz) +
               w[2].x * cub_fetch(keys, values, voxel_count, idx.x + 2, idx.y + j, idx.z + k, dimx, dimy, dimz) +
               w[3].x * cub_fetch(keys, values, voxel_count, idx.x + 3, idx.y + j, idx.z + k, dimx, dimy, dimz));
        }
        result += w[k].z * col_acc;
      }
      return make_float4(result, result, result, 1.0f);
    }

    /* INTERPOLATION_CLOSEST */
    const int ix = (int)(floorf(px));
    const int iy = (int)(floorf(py));
    const int iz = (int)(floorf(pz));
    const float f = cub_fetch(keys, values, voxel_count, ix, iy, iz, dimx, dimy, dimz);
    return make_float4(f, f, f, 1.0f);
  }
#else
  (void)kg;
  (void)sd;
  (void)image_texture_id;
  (void)P;
  (void)interp;
  (void)stochastic;
#endif

  return IMAGE_MISSING_RGBA;
}

#ifndef __KERNEL_GPU__
} /* Namespace. */
#endif

CCL_NAMESPACE_END
