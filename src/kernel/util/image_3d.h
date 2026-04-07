/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/sample/lcg.h"

#include "util/types_image.h"

#if !defined(__KERNEL_METAL__) && !defined(__KERNEL_ONEAPI__)
#  ifdef WITH_NANOVDB
#  include "kernel/util/nanovdb.h"
#  include <nanovdb/NanoVDB.h>
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
// NanoVDB Derivative Bundle Format - On-Device Structures
// ============================================================================
// Binary layout:
//   1. FileHeader (64 bytes, aligned to 32)
//   2. LevelHeader[levelCount]
//   3. GridHeader[gridCount]
//   4. NanoVDB grid payloads (each 32-byte aligned)
// ============================================================================

struct DerivFileHeader {
    uint32_t magic;              // Magic number: 0x4E56444D
    uint32_t version;            // File format version
    uint32_t payloadAlignment;   // Alignment requirement for grid payloads
    uint32_t levelCount;         // Number of levels
    uint32_t gridCount;          // Total number of grids
    uint32_t reserved1;
    uint64_t levelTableOffset;   // Byte offset to LevelHeader array
    uint64_t gridTableOffset;    // Byte offset to GridHeader array
    uint64_t payloadBlockOffset; // Byte offset to first grid payload
    uint64_t totalFileSize;      // Total file size in bytes
    uint64_t reserved2;
};

struct DerivLevelHeader {
    uint32_t levelIndex;         // Level index (0-based)
    uint32_t derivativeCount;    // Number of derivatives in this level
    uint32_t firstGridIndex;     // Index of first grid in GridHeader array
    uint32_t reserved;
};

struct DerivGridHeader {
    uint32_t levelIndex;                // Which level this grid belongs to
    uint32_t derivativeIndex;           // Derivative index within level (0-based)
    uint32_t derivativeCountInLevel;    // Total derivatives in this level
    uint32_t reserved1;
    uint64_t payloadOffset;             // Byte offset to grid payload
    uint64_t payloadSize;               // Size of grid payload in bytes
    int32_t  bboxMin[3];                // Bounding box min
    int32_t  bboxMax[3];                // Bounding box max
    uint32_t dims[3];                   // Grid dimensions
    uint32_t reserved2;
    char     name[56];                  // Grid name for debugging
};

// ============================================================================
// Helpers for NanoVDB Derivative Bundle
// ============================================================================

// Taylor polynomial basis functions
// Maps derivative index to basis monomial for Taylor series reconstruction
// Given local offset (px, py, pz) from voxel center, returns the basis value
// Coefficients include factorial terms from Taylor expansion: f(x) = Σ (∂^n f / ∂x^n) * x^n / n!
ccl_device_inline double derivBasisValue(int derivIdx, double px, double py, double pz)
{
    switch (derivIdx) {
        // 0th order: constant term
        case 0:  return 1.0;
        
        // 1st order: linear terms
        case 1:  return px;
        case 2:  return py;
        case 3:  return pz;
        
        // 2nd order: pure quadratic terms
        // Divided by 2! = 2
        case 4:  return px * px * 0.5;
        case 5:  return py * py * 0.5;
        case 6:  return pz * pz * 0.5;
        
        // 2nd order: mixed terms
        case 7:  return px * py;
        case 8:  return px * pz;
        case 9:  return py * pz;
        
        // 3rd order: pure cubic terms
        // Divided by 3! = 6
        case 10: return px * px * px * (1.0 / 6.0);
        case 11: return py * py * py * (1.0 / 6.0);
        case 12: return pz * pz * pz * (1.0 / 6.0);
        
        // 3rd order: mixed terms
        // Divided by 2! for the squared term
        case 13: return px * px * py * 0.5;
        case 14: return px * px * pz * 0.5;
        case 15: return py * py * px * 0.5;
        case 16: return py * py * pz * 0.5;
        case 17: return pz * pz * px * 0.5;
        case 18: return pz * pz * py * 0.5;
        
        // 3rd order: fully mixed term
        case 19: return px * py * pz;
        
        // 4th order: pure quartic terms
        // Divided by 4! = 24
        case 20: return px * px * px * px / 24.0;
        case 21: return py * py * py * py / 24.0;
        case 22: return pz * pz * pz * pz / 24.0;
        
        // 4th order: mixed cubic-linear terms
        // Divided by 3! = 6 for the cubic term
        case 23: return px * px * px * py / 6.0;
        case 24: return px * px * px * pz / 6.0;
        case 25: return px * py * py * px / 6.0;
        case 26: return py * py * py * pz / 6.0;
        case 27: return pz * pz * pz * px / 6.0;
        case 28: return pz * pz * pz * py / 6.0;
        
        // 4th order: mixed quadratic-quadratic terms
        // Divided by 2! * 2! = 4
        case 29: return px * px * py * py / 4.0;
        case 30: return px * px * pz * pz / 4.0;
        case 31: return py * py * pz * pz / 4.0;
        
        // 4th order: mixed quadratic-linear-linear terms
        // Divided by 2! = 2 for the squared term
        case 32: return px * px * py * pz / 2.0;
        case 33: return py * py * px * pz / 2.0;
        case 34: return pz * pz * px * py / 2.0;
        
        default: return 0.0;
    }
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

    const float wx = x, wy = y, wz = z;

    // Get voxel size from level 0 (finest level) for consistent normalization
    const DerivLevelHeader& level0Header = levelTable[0];
    const DerivGridHeader& level0Grid0Header = gridTable[level0Header.firstGridIndex];
    const ccl_global nanovdb::NanoGrid<T>* level0Grid0 = getDerivGridPtr<T>(base, level0Grid0Header);
    const double level0_voxel_size0 = level0Grid0->voxelSize()[0];
    const double level0_voxel_size1 = level0Grid0->voxelSize()[1];
    const double level0_voxel_size2 = level0Grid0->voxelSize()[2];

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

        // Local offset from voxel center for Taylor expansion
        // Calculate voxel center in index space
        const nanovdb::Vec3d voxel_center_idx(ix + 0.5, iy + 0.5, iz + 0.5);
        // Convert to world coordinates
        const nanovdb::Vec3d voxel_center_world = grid0->indexToWorld(voxel_center_idx);
        // Calculate offset from voxel center in world space, normalized by level 0 voxel size
        const double px = (wx - voxel_center_world[0]) / level0_voxel_size0;
        const double py = (wy - voxel_center_world[1]) / level0_voxel_size1;
        const double pz = (wz - voxel_center_world[2]) / level0_voxel_size2;

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

#endif /* WITH_NANOVDB */

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
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_EMPTY) {
    return zero_float4();
  }
  if (data_type == IMAGE_DATA_TYPE_RAW3D_FLOAT) {
    float* data = (float *)info.data;
        float w = tex.transform_3d.x.x;
        float h = tex.transform_3d.y.y;

    size_t index = (size_t)floorf(P.x) + (size_t)floorf(P.y) * w + (size_t)floorf(P.z) * w * h;
    const float f = data[index];
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_RAW3D_FLOAT3) {
    float3* data = (float3 *)info.data;
        float w = tex.transform_3d.x.x;
        float h = tex.transform_3d.y.y;

    size_t index = (size_t)floorf(P.x) + (size_t)floorf(P.y) * w + (size_t)floorf(P.z) * w * h;
    const float3 f = data[index];
    return make_float4(f, 1.0f);
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
