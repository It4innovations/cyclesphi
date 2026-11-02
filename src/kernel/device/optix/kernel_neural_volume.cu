/* SPDX-FileCopyrightText: 2011-2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Volume shading as OptiX programs for the COOPVEC backend of neural volumes (SRNXR format 2,
 * kernel/util/image_3d_srnxr_decode.h): OptiX cooperative vectors are only available inside
 * OptiX programs, so the kernels that evaluate volume shaders run as ray generation programs
 * of this module when a scene uses the backend (OptiXDevice::use_neural_coopvec). SVM shading,
 * loaded as a separate module like the shader ray-tracing kernels. Needs OptiX 9.0+ headers. */

#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 1
#define SRNXR_COOPVEC_DIRECT_CALLABLES 1

#include "kernel/device/optix/compat.h"
#include "kernel/device/optix/globals.h"

#include "kernel/device/gpu/image.h"

#include "kernel/tables.h"

#include "kernel/integrator/state.h"
#include "kernel/integrator/state_flow.h"
#include "kernel/integrator/state_util.h"

#include "kernel/film/data_passes.h"

#include "kernel/bake/bake.h"
#include "kernel/integrator/shade_shadow.h"
#include "kernel/integrator/shade_volume.h"

#ifdef SRNXR_HAS_COOPVEC
/* Cooperative vector network evaluation, direct callables 0, 1, 2 of the volume programs. */
extern "C" __device__ SRNXRCoopVecOut __direct_callable__srnxr_coopvec_64(
    const ccl_global uint8_t *base, uint32_t network, float x0, float x1, float x2, float x3)
{
  return srnxr_coopvec_callable_body<64>(base, network, x0, x1, x2, x3);
}

extern "C" __device__ SRNXRCoopVecOut __direct_callable__srnxr_coopvec_128(
    const ccl_global uint8_t *base, uint32_t network, float x0, float x1, float x2, float x3)
{
  return srnxr_coopvec_callable_body<128>(base, network, x0, x1, x2, x3);
}

extern "C" __device__ SRNXRCoopVecOut __direct_callable__srnxr_coopvec_256(
    const ccl_global uint8_t *base, uint32_t network, float x0, float x1, float x2, float x3)
{
  return srnxr_coopvec_callable_body<256>(base, network, x0, x1, x2, x3);
}
#endif

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_volume()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_volume(nullptr, path_index, kernel_params.render_buffer);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_volume_ray_marching()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_volume_ray_marching(nullptr, path_index, kernel_params.render_buffer);
}

extern "C" __global__ void __raygen__kernel_optix_integrator_shade_shadow()
{
  const int global_index = optixGetLaunchIndex().x;
  const int path_index = (kernel_params.path_index_array) ?
                             kernel_params.path_index_array[global_index] :
                             global_index;
  integrator_shade_shadow(nullptr, path_index, kernel_params.render_buffer);
}

extern "C" __global__ void __raygen__kernel_optix_shader_eval_volume_density()
{
  KernelShaderEvalInput *const input = (KernelShaderEvalInput *)kernel_params.path_index_array;
  float *const output = kernel_params.render_buffer;
  uint *const cache_miss = kernel_params.shader_eval_cache_miss;
  const int global_index = kernel_params.shader_eval_offset + optixGetLaunchIndex().x;
  kernel_volume_density_evaluate(nullptr, input, output, cache_miss, global_index);
}
