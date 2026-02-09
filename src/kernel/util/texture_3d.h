/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/globals.h"
#include "kernel/sample/lcg.h"

#include "util/texture.h"

#if !defined(__KERNEL_METAL__) && !defined(__KERNEL_ONEAPI__)
#  ifdef WITH_NANOVDB
//#    include "kernel/util/nanovdb.h"
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
ccl_device_inline void fill_cubic_weights(float3 w[4], float3 t)
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
ccl_device OutT kernel_tex_image_interp_trilinear_nanovdb(ccl_private Acc &acc, const float3 P)
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
ccl_device OutT kernel_tex_image_interp_tricubic_nanovdb(ccl_private Acc &acc, const float3 P)
{
  const float3 floor_P = floor(P);
  const float3 t = P - floor_P;
  const int3 index = make_int3(floor_P) - make_int3(1);

  float3 w[4];
  fill_cubic_weights(w, t);

  OutT result = make_zero<OutT>();

  for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 4; j++) {
      result += w[k].z * (w[j].y * (w[0].x * (OutT(acc.getValue(index + make_int3(0, j, k)))) +
                                    w[1].x * (OutT(acc.getValue(index + make_int3(1, j, k)))) +
                                    w[2].x * (OutT(acc.getValue(index + make_int3(2, j, k)))) +
                                    w[3].x * (OutT(acc.getValue(index + make_int3(3, j, k))))));
    }
  }

  return result;
}

template<typename OutT, typename T>
#  if defined(__KERNEL_METAL__)
__attribute__((noinline))
#  else
ccl_device_noinline
#  endif
OutT kernel_tex_image_interp_nanovdb(const ccl_global TextureInfo &info,
                                     float3 P,
                                     const InterpolationType interp)
{
  ccl_global nanovdb::NanoGrid<T> *const grid = (ccl_global nanovdb::NanoGrid<T> *)info.data;

  // if (interp == INTERPOLATION_CLOSEST) {
  //   nanovdb::ReadAccessor<T> acc(grid->tree().root());
  //   return OutT(acc.getValue(make_int3(floor(P))));
  // }

  // nanovdb::CachedReadAccessor<T> acc(grid->tree().root());
  // if (interp == INTERPOLATION_LINEAR) {
  //   return kernel_tex_image_interp_trilinear_nanovdb<OutT>(acc, P);
  // }

  // return kernel_tex_image_interp_tricubic_nanovdb<OutT>(acc, P);

  // INTERPOLATION_CLOSEST
  nanovdb::ReadAccessor<T> acc(grid->tree().root());
  const nanovdb::Coord coord((int32_t)floorf(P.x), (int32_t)floorf(P.y), (int32_t)floorf(P.z));
  return OutT(acc.getValue(coord));
}


#  if defined(__KERNEL_METAL__)
template<typename OutT, typename T>
__attribute__((noinline)) OutT kernel_tex_image_interp_nanovdb_multires(const ccl_global TextureInfo &info,
                                                               const float x,
                                                               const float y,
                                                               const float z,
                                                               const uint interpolation)
#  else
template<typename OutT, typename T>
ccl_device_noinline OutT kernel_tex_image_interp_nanovdb_multires(const ccl_global TextureInfo &info,
                                                         const float x,
                                                         const float y,
                                                         const float z,
                                                         const uint interpolation)
#  endif
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
        if (i < levels - 1) {
            // Read next grid offset
            size_t next_offset = *((size_t*)((char*)info.data + offset));
            // Grid data starts after the offset field
            ccl_global NanoGrid<T>* const grid = (ccl_global NanoGrid<T>*)((char*)info.data + (offset + sizeof(size_t)));

            nanovdb::Vec3d coord_index = grid->worldToIndex(nanovdb::Vec3d(x, y, z));

            ReadAccessor<T> acc(grid->tree().root());
            const nanovdb::Coord coord((int32_t)floorf(coord_index[0]), (int32_t)floorf(coord_index[1]), (int32_t)floorf(coord_index[2]));
            OutT f = acc.getValue(coord);

            bool is_nonzero = false;
            if constexpr (sizeof(OutT) == sizeof(float)) {
                is_nonzero = (f != 0.0f);
            }
            else {
                // For vector types, check if any component is non-zero
                is_nonzero = (f.x != 0.0f || f.y != 0.0f || f.z != 0.0f);
            }

            if (is_nonzero)
                return f;

            // Jump to next offset position
            offset = next_offset - sizeof(size_t);
        }
        else {
            // Last grid has no offset field
            ccl_global NanoGrid<T>* const grid = (ccl_global NanoGrid<T>*)((char*)info.data + offset);

            nanovdb::Vec3d coord_index = grid->worldToIndex(nanovdb::Vec3d(x, y, z));

            ReadAccessor<T> acc(grid->tree().root());
            const nanovdb::Coord coord((int32_t)floorf(coord_index[0]), (int32_t)floorf(coord_index[1]), (int32_t)floorf(coord_index[2]));
            OutT f = acc.getValue(coord);

            bool is_nonzero = false;
            if constexpr (sizeof(OutT) == sizeof(float)) {
                is_nonzero = (f != 0.0f);
            }
            else {
                // For vector types, check if any component is non-zero
                is_nonzero = (f.x != 0.0f || f.y != 0.0f || f.z != 0.0f);
            }

            if (is_nonzero)
                return f;
        }
    }    

    return OutT(0.0f);    
}

#endif /* WITH_NANOVDB */

ccl_device float4 kernel_tex_image_interp_3d(KernelGlobals kg,
                                             ccl_private ShaderData *sd,
                                             const int id,
                                             float3 P,
                                             InterpolationType interp,
                                             const bool stochastic)
{
#ifdef WITH_NANOVDB
  const ccl_global TextureInfo &info = kernel_data_fetch(texture_info, id);

  if (info.use_transform_3d) {
    P = transform_point(&info.transform_3d, P);
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
    const float f = kernel_tex_image_interp_nanovdb<float, float>(info, P, interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FLOAT3) {
    const float3 f = kernel_tex_image_interp_nanovdb<float3, packed_float3>(
        info, P, interpolation);
    return make_float4(f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FLOAT4) {
    return kernel_tex_image_interp_nanovdb<float4, float4>(info, P, interpolation);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FPN) {
    const float f = kernel_tex_image_interp_nanovdb<float, nanovdb::FpN>(info, P, interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_FP16) {
    const float f = kernel_tex_image_interp_nanovdb<float, nanovdb::Fp16>(info, P, interpolation);
    return make_float4(f, f, f, 1.0f);
  }
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT) {
    const float f = kernel_tex_image_interp_nanovdb_multires<float, float>(info, P.x, P.y, P.z, (const uint)interpolation);
    return make_float4(f, f, f, 1.0f);
  }  
  if (data_type == IMAGE_DATA_TYPE_NANOVDB_EMPTY) {
    return zero_float4();
  }
#else
  (void)kg;
  (void)sd;
  (void)id;
  (void)P;
  (void)interp;
  (void)stochastic;
#endif

  return make_float4(
      TEX_IMAGE_MISSING_R, TEX_IMAGE_MISSING_G, TEX_IMAGE_MISSING_B, TEX_IMAGE_MISSING_A);
}

#ifndef __KERNEL_GPU__
} /* Namespace. */
#endif

CCL_NAMESPACE_END
