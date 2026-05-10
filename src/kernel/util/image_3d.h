/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/sample/lcg.h"

#include "util/types_image.h"
#include "kernel/util/image_3d_derivates.h"

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
ccl_device_inline double derivBasisValue(int basis, double p_x, double p_y, double p_z)
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
ccl_device_inline const ccl_global nanovdb::NanoGrid<T>* getDerivGridPtr(
    const uint8_t* base, 
    const DerivGridHeader& gh)
{
    return reinterpret_cast<const ccl_global nanovdb::NanoGrid<T>*>(base + gh.payloadOffset);
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
    using namespace nanovdb;

    // Format description of bin file:
    // size_t : number of levels (aligned to 32 bytes)
    // size_t : offset to grid1
    // grid0 data (aligned to 32 bytes)
    // size_t : offset to grid2
    // grid1 data (aligned to 32 bytes)
    // ...

    size_t offset = 0;
    // Read number of levels
    size_t levels = *((size_t*)((char*)info.data + offset));
    // Align to 32 bytes after num_levels
    offset = 32;

    for (size_t i = 0; i < levels; ++i) {
        // Get pointer to current grid data        
        const char* grid_ptr = nullptr;
        size_t next_offset = 0;

        if (i < levels - 1) {
            // Read next grid offset
            next_offset = *((size_t*)((char*)info.data + offset));
            // Grid data starts after the offset field
            grid_ptr = (char*)info.data + (offset + sizeof(size_t));
        } else {
            // Last grid has no offset field
            grid_ptr = (char*)info.data + offset;
        }

        const ccl_global NanoGrid<T>* const grid = 
            reinterpret_cast<const ccl_global NanoGrid<T>*>(grid_ptr);

        nanovdb::Vec3d coord_index = grid->worldToIndex(nanovdb::Vec3d(x, y, z));

        ReadAccessor<T> acc(grid->tree().root());
        const nanovdb::Coord coord((int32_t)floorf((float)coord_index[0]), 
                                    (int32_t)floorf((float)coord_index[1]), 
                                    (int32_t)floorf((float)coord_index[2]));
        OutT f = acc.getValue(coord);

        bool is_nonzero = false;
        if constexpr (sizeof(OutT) == sizeof(float)) {
            is_nonzero = (f != 0.0f);
        }
        else {
            // For vector types, check if any component is non-zero
            is_nonzero = (f.x != 0.0f || f.y != 0.0f || f.z != 0.0f);
        }

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
    using namespace nanovdb;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(info.data);

    // Read FileHeader
    const DerivFileHeader* fh = reinterpret_cast<const DerivFileHeader*>(base);
    
    // Validate magic number (optional, can be removed for performance)
    // if (fh->magic != 0x4E56444D) return OutT(0.0f);

    const uint32_t levelCount = fh->levelCount;
    if (levelCount == 0) return OutT(0.0f);

    const DerivLevelHeader* levelTable = 
        reinterpret_cast<const DerivLevelHeader*>(base + fh->levelTableOffset);
    const DerivGridHeader* gridTable = 
        reinterpret_cast<const DerivGridHeader*>(base + fh->gridTableOffset);

    const double wx = x, wy = y, wz = z;

    // Iterate through levels (coarsest to finest logic, or customize as needed)
    // Here we check each level for non-zero derivatives and reconstruct
    for (uint32_t levelIdx = 0; levelIdx < levelCount; ++levelIdx) {
        const DerivLevelHeader& lh = levelTable[levelIdx];
        
        const uint32_t derivCount = lh.derivativeCount;
        const uint32_t firstGrid = lh.firstGridIndex;

        // Bounds check
        if (firstGrid + derivCount > fh->gridCount) continue;

        // Sample first grid to get reference point for local coordinates
        // All grids in a level share the same index space
        const DerivGridHeader& gh0 = gridTable[firstGrid];
        const ccl_global nanovdb::NanoGrid<T>* grid0 = getDerivGridPtr<T>(base, gh0);
        
        // Convert world to index space
        const nanovdb::Vec3d ijk_d = grid0->worldToIndex(nanovdb::Vec3d(wx, wy, wz));
        
        // Integer voxel coordinate
        const int32_t ix = (int32_t)floorf((float)ijk_d[0]);
        const int32_t iy = (int32_t)floorf((float)ijk_d[1]);
        const int32_t iz = (int32_t)floorf((float)ijk_d[2]);
        const nanovdb::Coord coord(ix, iy, iz);

        // Get current level's voxel size for proper normalization
        const double current_voxel_size0 = grid0->voxelSize()[0];
        const double current_voxel_size1 = grid0->voxelSize()[1];
        const double current_voxel_size2 = grid0->voxelSize()[2];

        // Local offset from voxel center for Taylor expansion
        // Calculate voxel center in index space
        const nanovdb::Vec3d voxel_center_idx(ix + 0.5, iy + 0.5, iz + 0.5);
        // Convert to world coordinates
        const nanovdb::Vec3d voxel_center_world = grid0->indexToWorld(voxel_center_idx);
        // Normalize by half the current level's voxel size to get [-1, 1] range
        // Following formula: p = (coo - center) / (gridSize / 2)
        const double px = (wx - voxel_center_world[0]) / (0.5 * current_voxel_size0);
        const double py = (wy - voxel_center_world[1]) / (0.5 * current_voxel_size1);
        const double pz = (wz - voxel_center_world[2]) / (0.5 * current_voxel_size2);

        // Accumulate Taylor polynomial reconstruction
        double result = 0.0;
        bool hasNonZero = false;

        for (uint32_t d = 0; d < derivCount; ++d) {
            const DerivGridHeader& gh = gridTable[firstGrid + d];
            const ccl_global nanovdb::NanoGrid<T>* grid = getDerivGridPtr<T>(base, gh);

            // Read coefficient from grid
            ReadAccessor<T> acc(grid->tree().root());
            const T coeff = acc.getValue(coord);

            // Check if coefficient is non-zero
            bool nonzero = false;
            if constexpr (sizeof(T) == sizeof(float)) {
                nonzero = (coeff != 0.0f);
            } else {
                // For vector types (if needed)
                nonzero = (coeff != T(0.0f));
            }

            if (nonzero) {
                hasNonZero = true;
                // Multiply coefficient by basis function and accumulate
                const double basis = derivBasisValue(gh.derivativeIndex, px, py, pz);
                result += (double)coeff * basis;
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
    using namespace nanovdb;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(info.data);

    // Read FileHeader
    const DerivFileHeader* fh = reinterpret_cast<const DerivFileHeader*>(base);
    
    const uint32_t levelCount = fh->levelCount;
    if (levelCount == 0) return OutT(0.0f);

    const DerivLevelHeader* levelTable = 
        reinterpret_cast<const DerivLevelHeader*>(base + fh->levelTableOffset);
    const DerivGridHeader* gridTable = 
        reinterpret_cast<const DerivGridHeader*>(base + fh->gridTableOffset);

    const double wx = x, wy = y, wz = z;

    // Iterate through levels
    for (uint32_t levelIdx = 0; levelIdx < levelCount; ++levelIdx) {
        const DerivLevelHeader& lh = levelTable[levelIdx];
        
        const uint32_t packedGridCount = lh.derivativeCount;  // Number of packed grids
        const uint32_t firstGrid = lh.firstGridIndex;

        // Bounds check
        if (firstGrid + packedGridCount > fh->gridCount) continue;

        // Sample first grid to get reference point for local coordinates
        const DerivGridHeader& gh0 = gridTable[firstGrid];
        const ccl_global nanovdb::NanoGrid<float>* grid0 = 
            reinterpret_cast<const ccl_global nanovdb::NanoGrid<float>*>(
                base + gh0.payloadOffset);
        
        // Convert world to index space
        const nanovdb::Vec3d ijk_d = grid0->worldToIndex(nanovdb::Vec3d(wx, wy, wz));
        
        // Integer voxel coordinate
        const int32_t ix = (int32_t)floorf((float)ijk_d[0]);
        const int32_t iy = (int32_t)floorf((float)ijk_d[1]);
        const int32_t iz = (int32_t)floorf((float)ijk_d[2]);
        const nanovdb::Coord coord(ix, iy, iz);

        // Get current level's voxel size for proper normalization
        const double current_voxel_size0 = grid0->voxelSize()[0];
        const double current_voxel_size1 = grid0->voxelSize()[1];
        const double current_voxel_size2 = grid0->voxelSize()[2];

        // Local offset from voxel center for Taylor expansion
        const nanovdb::Vec3d voxel_center_idx(ix + 0.5, iy + 0.5, iz + 0.5);
        const nanovdb::Vec3d voxel_center_world = grid0->indexToWorld(voxel_center_idx);
        // Normalize by half the current level's voxel size to get [-1, 1] range
        // Following formula: p = (coo - center) / (gridSize / 2)
        const double px = (wx - voxel_center_world[0]) / (0.5 * current_voxel_size0);
        const double py = (wy - voxel_center_world[1]) / (0.5 * current_voxel_size1);
        const double pz = (wz - voxel_center_world[2]) / (0.5 * current_voxel_size2);

        // Accumulate Taylor polynomial reconstruction
        double result = 0.0;
        bool hasNonZero = false;

        // Iterate through packed grids in this level
        for (uint32_t packedIdx = 0; packedIdx < packedGridCount; ++packedIdx) {
            const DerivGridHeader& gh = gridTable[firstGrid + packedIdx];
            
            // Base derivative index for this packed grid
            const uint32_t baseDerivIdx = packedIdx * 4;

            // Dispatch based on grid type
            const DerivGridType gridType = (DerivGridType)gh.gridType;
            
            if (gridType == DERIV_GRID_TYPE_FLOAT) {
                // Float grid - 1 component
                const ccl_global nanovdb::NanoGrid<float>* grid = 
                    reinterpret_cast<const ccl_global nanovdb::NanoGrid<float>*>(
                        base + gh.payloadOffset);
                ReadAccessor<float> acc(grid->tree().root());
                const float coeff = acc.getValue(coord);
                
                if (coeff != 0.0f) {
                    hasNonZero = true;
                    const double basis = derivBasisValue(baseDerivIdx, px, py, pz);
                    result += (double)coeff * basis;
                }
            }
            else if (gridType == DERIV_GRID_TYPE_VEC3F) {
                // Vec3f grid - up to 3 components
                const ccl_global nanovdb::NanoGrid<packed_float3>* grid = 
                    reinterpret_cast<const ccl_global nanovdb::NanoGrid<packed_float3>*>(
                        base + gh.payloadOffset);
                ReadAccessor<packed_float3> acc(grid->tree().root());
                const packed_float3 packedValue = acc.getValue(coord);
                
                for (uint32_t compIdx = 0; compIdx < 3; ++compIdx) {
                    const float coeff = readPackedComponent(packedValue, compIdx);
                    if (coeff != 0.0f) {
                        hasNonZero = true;
                        const uint32_t derivIdx = baseDerivIdx + compIdx;
                        const double basis = derivBasisValue(derivIdx, px, py, pz);
                        result += (double)coeff * basis;
                    }
                }
            }
            else if (gridType == DERIV_GRID_TYPE_VEC4F) {
                // Vec4f grid - 4 components
                const ccl_global nanovdb::NanoGrid<float4>* grid = 
                    reinterpret_cast<const ccl_global nanovdb::NanoGrid<float4>*>(
                        base + gh.payloadOffset);
                ReadAccessor<float4> acc(grid->tree().root());
                const float4 packedValue = acc.getValue(coord);
                
                for (uint32_t compIdx = 0; compIdx < 4; ++compIdx) {
                    const float coeff = readPackedComponent(packedValue, compIdx);
                    if (coeff != 0.0f) {
                        hasNonZero = true;
                        const uint32_t derivIdx = baseDerivIdx + compIdx;
                        const double basis = derivBasisValue(derivIdx, px, py, pz);
                        result += (double)coeff * basis;
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

  InterpolationType interpolation = (interp == INTERPOLATION_NONE) ?
                                        (InterpolationType)info.interpolation :
                                        interp;

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

    float scale_x = tex.transform_3d.x.y;
    float scale_y = tex.transform_3d.y.y;
    float scale_z = tex.transform_3d.z.y;

    float trans_x = tex.transform_3d.x.z;
    float trans_y = tex.transform_3d.y.z;
    float trans_z = tex.transform_3d.z.z;

    float bbox_min_x = tex.transform_3d.x.w;
    float bbox_min_y = tex.transform_3d.y.w;
    float bbox_min_z = tex.transform_3d.z.w;

    float px = (P.x - bbox_min_x - trans_x) / scale_x;
    float py = (P.y - bbox_min_y - trans_y) / scale_y;
    float pz = (P.z - bbox_min_z - trans_z) / scale_z;

    if (px < 0.0f || py < 0.0f || pz < 0.0f)
      return zero_float4();

    if (floorf(px) >= dimx || 
        floorf(py) >= dimy || 
        floorf(pz) >= dimz) {
      return zero_float4();
    }

    float* data = (float *)info.data;
    
    // P is in index space after identity transform - compute array index
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

    float scale_x = tex.transform_3d.x.y;
    float scale_y = tex.transform_3d.y.y;
    float scale_z = tex.transform_3d.z.y;

    float trans_x = tex.transform_3d.x.z;
    float trans_y = tex.transform_3d.y.z;
    float trans_z = tex.transform_3d.z.z;

    float bbox_min_x = tex.transform_3d.x.w;
    float bbox_min_y = tex.transform_3d.y.w;
    float bbox_min_z = tex.transform_3d.z.w;

    float px = (P.x - bbox_min_x - trans_x) / scale_x;
    float py = (P.y - bbox_min_y - trans_y) / scale_y;
    float pz = (P.z - bbox_min_z - trans_z) / scale_z;

    if (px < 0.0f || py < 0.0f || pz < 0.0f)
      return zero_float4();

    if (floorf(px) >= dimx || floorf(py) >= dimy || floorf(pz) >= dimz) {
      return zero_float4();
    }

    // P is in index space after identity transform - compute array index
    const size_t ix = (size_t)(floorf(px));
    const size_t iy = (size_t)(floorf(py));
    const size_t iz = (size_t)(floorf(pz));

    const SerializableZFPData* __restrict__ header =
        (const SerializableZFPData*)info.data;

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

    const size_t bx = ix / 4;
    const size_t by = iy / 4;
    const size_t bz = iz / 4;
    const size_t block_idx = bx + header->block_dims_x * (by + header->block_dims_y * bz);
    const uint local_idx = (ix & 3) + 4 * ((iy & 3) + 4 * (iz & 3));

    cuZFP::NoCache<float, 64> cache;
    const float f = cache.get(store, block_idx, local_idx);
    return make_float4(f, f, f, 1.0f);
  }
#endif
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
