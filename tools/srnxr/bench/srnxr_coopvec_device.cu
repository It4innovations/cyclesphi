/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* OptiX programs of srnxr_coopvec: one raygen launch index per query, the Cycles SRNXR
 * decoder with the COOPVEC network evaluator (kernel/util/image_3d_srnxr_decode.h). */

#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 1
#include <optix.h>

#include <cuda_fp16.h>

#define __KERNEL_OPTIX__
#include "srnxr_compat.h"

#include "srnxr_coopvec_params.h"

using namespace ccl;

extern "C" {
__constant__ SRNXRCoopVecParams params;
}

extern "C" __global__ void __raygen__srnxr()
{
  const int i = int(optixGetLaunchIndex().x);
  if (i >= params.n) {
    return;
  }
  const float4 q = params.points[i];
  const float p[4] = {q.x, q.y, q.z, q.w};
  float v[SRNXR_V2_MAX_OUTPUTS];
  srnxr_v2_decode(params.blob, p, true, v);
  for (int k = 0; k < params.output_dim; k++) {
    params.out[size_t(i) * params.output_dim + k] = v[k];
  }
}

extern "C" __global__ void __miss__srnxr() {}
