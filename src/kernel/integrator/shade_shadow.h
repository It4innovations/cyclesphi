/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/shade_volume.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/integrator/volume_stack.h"

#include "kernel/geom/shader_data.h"
#include "kernel/light/light.h"

CCL_NAMESPACE_BEGIN

enum TransparentShadowEvalResult {
  TRANSPARENT_SHADOW_EVAL_CONTINUE = 0,
  TRANSPARENT_SHADOW_EVAL_OPAQUE = 1,
  TRANSPARENT_SHADOW_EVAL_CACHE_MISS = 2,
};

#ifdef __KERNEL_GPU__
/* Assume we can pack num hits into 12 bits, so that we can also store resume hits
 * and skip volume in the 16 bit num_hits. */
#  define SHADOW_HIT_COUNT_BITS_GPU 12
#  define SHADOW_HIT_COUNT_MASK_GPU ((1 << SHADOW_HIT_COUNT_BITS_GPU) - 1)
#  define SHADOW_RESUME_BITS_GPU 3
#  define SHADOW_RESUME_MASK_GPU ((1 << SHADOW_RESUME_BITS_GPU) - 1)

/* +1 is for the extra loop iteration for the final volume segment. */
static_assert(INTEGRATOR_SHADOW_ISECT_SIZE_GPU + 1 < (1 << SHADOW_RESUME_BITS_GPU),
              "INTEGRATOR_SHADOW_ISECT_SIZE_GPU too large for resume_hit bits");
#endif

ccl_device_inline uint shadow_num_hits_get(const uint packed_num_hits)
{
#ifdef __KERNEL_GPU__
  return packed_num_hits & SHADOW_HIT_COUNT_MASK_GPU;
#else
  return packed_num_hits;
#endif
}

ccl_device_inline uint shadow_resume_hit_get(const uint packed_num_hits)
{
#ifdef __KERNEL_GPU__
  return (packed_num_hits >> SHADOW_HIT_COUNT_BITS_GPU) & SHADOW_RESUME_MASK_GPU;
#else
  (void)packed_num_hits;
  return 0;
#endif
}

ccl_device_inline bool shadow_skip_volume_get(const uint packed_num_hits)
{
#ifdef __KERNEL_GPU__
  return (packed_num_hits >> (SHADOW_HIT_COUNT_BITS_GPU + SHADOW_RESUME_BITS_GPU)) & 1;
#else
  (void)packed_num_hits;
  return false;
#endif
}

ccl_device_inline uint shadow_num_hits_pack(const uint num_hits,
                                            const uint resume_hit,
                                            const bool skip_volume)
{
#ifdef __KERNEL_GPU__
  return (num_hits & SHADOW_HIT_COUNT_MASK_GPU) |
         ((resume_hit & SHADOW_RESUME_MASK_GPU) << SHADOW_HIT_COUNT_BITS_GPU) |
         ((skip_volume ? 1u : 0u) << (SHADOW_HIT_COUNT_BITS_GPU + SHADOW_RESUME_BITS_GPU));
#else
  /* Cache miss resume not supported on CPU. */
  kernel_assert(resume_hit == 0 && !skip_volume);
  (void)resume_hit;
  (void)skip_volume;
  return num_hits;
#endif
}

ccl_device_inline bool shadow_intersections_has_remaining(const uint packed_num_hits)
{
  return shadow_num_hits_get(packed_num_hits) >= INTEGRATOR_SHADOW_ISECT_SIZE;
}

#ifdef __TRANSPARENT_SHADOWS__
ccl_device_inline Spectrum
integrate_transparent_surface_shadow(KernelGlobals kg,
                                     IntegratorShadowState state,
                                     const int hit,
                                     ccl_private ShaderEvalResult &result,
                                     const bool neural_record = false)
{
  PROFILING_INIT(kg, PROFILING_SHADE_SHADOW_SURFACE);

  /* TODO: does aliasing like this break automatic SoA in CUDA?
   * Should we instead store closures separate from ShaderData?
   *
   * TODO: is it better to declare this outside the loop or keep it local
   * so the compiler can see there is no dependency between iterations? */
  ShaderDataTinyStorage shadow_sd_storage;
  ccl_private ShaderData *shadow_sd = AS_SHADER_DATA(&shadow_sd_storage);

  /* Setup shader data at surface. */
  Intersection isect ccl_optional_struct_init;
  integrator_state_read_shadow_isect(state, &isect, hit);

  Ray ray ccl_optional_struct_init;
  integrator_state_read_shadow_ray(state, &ray);

  shader_setup_from_ray(kg, shadow_sd, &ray, &isect);

  /* Evaluate shader. When recording the neural volume lookups the surface counts as fully
   * transparent (the recorded walk covers at least the segments of the real one). */
  if (neural_record) {
    /* Pass. */
  }
  else if (!(shadow_sd->shader_flag & SD_HAS_ONLY_VOLUME)) {
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
        kg, state, shadow_sd, nullptr, PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_FLAG_NONE);
    if (shadow_sd->runtime_flag & SR_CACHE_MISS) {
      result = SHADER_EVAL_CACHE_MISS;
      return zero_spectrum();
    }
  }
  else {
    INTEGRATOR_STATE_WRITE(state, shadow_path, volume_bounds_bounce) += 1;
  }

#  ifdef __VOLUME__
  /* Exit/enter volume. */
  volume_stack_enter_exit<true>(kg, state, shadow_sd);
#  endif

  /* Disable transparent shadows for ray portals */
  if (shadow_sd->runtime_flag & SR_RAY_PORTAL) {
    result = SHADER_EVAL_EMPTY;
    return zero_spectrum();
  }

  /* Compute transparency from closures. */
  result = SHADER_EVAL_OK;
  return neural_record ? one_spectrum() : surface_shader_transparency(shadow_sd);
}

#  ifdef __VOLUME__
ccl_device_inline bool integrate_transparent_volume_shadow(KernelGlobals kg,
                                                           IntegratorShadowState state,
                                                           const int hit,
                                                           const int num_recorded_hits,
                                                           ccl_private Spectrum *ccl_restrict
                                                               throughput,
                                                           const uint neural_mode,
                                                           const uint neural_slot,
                                                           ccl_private uint *neural_cursor,
                                                           const uint neural_count)
{
  PROFILING_INIT(kg, PROFILING_SHADE_SHADOW_VOLUME);

  /* TODO: deduplicate with surface, or does it not matter for memory usage? */
  ShaderDataTinyStorage shadow_sd_storage;
  ccl_private ShaderData *shadow_sd = AS_SHADER_DATA(&shadow_sd_storage);

  /* Setup shader data. */
  Ray ray ccl_optional_struct_init;
  integrator_state_read_shadow_ray(state, &ray);
  ray.self.object = OBJECT_NONE;
  ray.self.prim = PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;
  /* Modify ray position and length to match current segment. */
  ray.tmin = (hit == 0) ? ray.tmin : INTEGRATOR_STATE_ARRAY(state, shadow_isect, hit - 1, t);
  ray.tmax = (hit < num_recorded_hits) ? INTEGRATOR_STATE_ARRAY(state, shadow_isect, hit, t) :
                                         ray.tmax;

  /* `object` is only needed for light tree with light linking, it is irrelevant for shadow. */
  shader_setup_from_volume(shadow_sd, &ray, OBJECT_NONE);
#  ifdef WITH_CYCLES_NEURAL_VOLUME
  shadow_sd->srnxr_queue_mode = neural_mode;
  shadow_sd->srnxr_queue_slot = neural_slot;
  shadow_sd->srnxr_queue_cursor = *neural_cursor;
  shadow_sd->srnxr_queue_count = neural_count;
#  else
  (void)neural_mode;
  (void)neural_slot;
  (void)neural_count;
#  endif

  if (kernel_data.integrator.volume_ray_marching) {
    const float step_size = volume_stack_step_size<true>(kg, state);
    volume_shadow_ray_marching(kg, state, &ray, shadow_sd, throughput, step_size);
  }
  else {
    volume_shadow_null_scattering(kg, state, &ray, shadow_sd, throughput);
  }
#  ifdef WITH_CYCLES_NEURAL_VOLUME
  *neural_cursor = shadow_sd->srnxr_queue_cursor;
#  endif

  return shadow_sd->runtime_flag & SR_CACHE_MISS;
}
#  endif

ccl_device_inline TransparentShadowEvalResult integrate_transparent_shadow(
    KernelGlobals kg,
    IntegratorShadowState state,
    const uint packed_num_hits,
    const uint neural_mode = 0,
    const uint neural_slot = 0,
    ccl_private uint *neural_cursor = nullptr,
    const uint neural_count = 0)
{
  uint neural_cursor_local = 0;
  if (neural_cursor == nullptr) {
    neural_cursor = &neural_cursor_local;
  }
  const bool neural_record = neural_mode == 1; /* SRNXR_QUEUE_RECORD */
  /* Accumulate shadow for transparent surfaces. */
  const uint num_hits = shadow_num_hits_get(packed_num_hits);
  const uint num_recorded_hits = min(num_hits, (uint)INTEGRATOR_SHADOW_ISECT_SIZE);

  /* Resume state from previous cache miss. */
  const uint resume_hit = shadow_resume_hit_get(packed_num_hits);
  const bool resume_skip_volume = shadow_skip_volume_get(packed_num_hits);

  /* Plus one to account for world volume, which has no boundary to hit but casts shadows. */
  for (uint hit = resume_hit; hit < num_recorded_hits + 1; hit++) {
    /* Skip volume if resuming after volume completed but surface had cache miss. */
    const bool skip_volume = (hit == resume_hit) && resume_skip_volume;

    /* Volume shaders. */
    if (!skip_volume &&
        (hit < num_recorded_hits || !shadow_intersections_has_remaining(packed_num_hits)))
    {
#  ifdef __VOLUME__
      if (!integrator_state_shadow_volume_stack_is_empty(kg, state)) {
        Spectrum throughput = INTEGRATOR_STATE(state, shadow_path, throughput);
        const bool cache_miss = integrate_transparent_volume_shadow(kg,
                                                                    state,
                                                                    hit,
                                                                    num_recorded_hits,
                                                                    &throughput,
                                                                    neural_mode,
                                                                    neural_slot,
                                                                    neural_cursor,
                                                                    neural_count);
        if (is_zero(throughput)) {
          return TRANSPARENT_SHADOW_EVAL_OPAQUE;
        }

        if (cache_miss) {
          /* Store resume state: restart at this hit, redo volume. */
          INTEGRATOR_STATE_WRITE(state, shadow_path, packed_num_hits) = shadow_num_hits_pack(
              num_hits, hit, false);
          return TRANSPARENT_SHADOW_EVAL_CACHE_MISS;
        }

        INTEGRATOR_STATE_WRITE(state, shadow_path, throughput) = throughput;
      }
#  endif
    }

    /* Surface shaders. */
    if (hit < num_recorded_hits) {
      ShaderEvalResult result = SHADER_EVAL_EMPTY;
      const Spectrum shadow = integrate_transparent_surface_shadow(
          kg, state, hit, result, neural_record);
      if (result == SHADER_EVAL_CACHE_MISS) {
        /* Store resume state: restart at this hit, skip volume. */
        INTEGRATOR_STATE_WRITE(state, shadow_path, packed_num_hits) = shadow_num_hits_pack(
            num_hits, hit, true);
        return TRANSPARENT_SHADOW_EVAL_CACHE_MISS;
      }

      const Spectrum throughput = INTEGRATOR_STATE(state, shadow_path, throughput) * shadow;
      if (is_zero(throughput)) {
        return TRANSPARENT_SHADOW_EVAL_OPAQUE;
      }

      INTEGRATOR_STATE_WRITE(state, shadow_path, throughput) = throughput;
      INTEGRATOR_STATE_WRITE(state, shadow_path, transparent_bounce) += 1;
      INTEGRATOR_STATE_WRITE(state, shadow_path, rng_offset) += PRNG_BOUNCE_NUM;
    }

    if (INTEGRATOR_STATE(state, shadow_path, volume_bounds_bounce) > VOLUME_BOUNDS_MAX) {
      return TRANSPARENT_SHADOW_EVAL_OPAQUE;
    }

    /* Note we do not need to check max_transparent_bounce here, the number
     * of intersections is already limited and made opaque in the
     * INTERSECT_SHADOW kernel. */
  }

  if (shadow_intersections_has_remaining(packed_num_hits)) {
    /* There are more hits that we could not recorded due to memory usage,
     * adjust ray to intersect again from the last hit. */
    const float last_hit_t = INTEGRATOR_STATE_ARRAY(state, shadow_isect, num_recorded_hits - 1, t);
    INTEGRATOR_STATE_WRITE(state, shadow_ray, tmin) = intersection_t_offset(last_hit_t);
  }

  return TRANSPARENT_SHADOW_EVAL_CONTINUE;
}
#endif /* __TRANSPARENT_SHADOWS__ */

#if defined(WITH_CYCLES_NEURAL_VOLUME) && defined(__TRANSPARENT_SHADOWS__) && \
    defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
/* Record the neural volume lookups of the transparent shadow walk of this path into the window
 * of its launch slot. The walk modifies the shadow path state (volume stack, random number
 * offset, throughput, ...), which is restored afterwards: the real walk at replay starts from the
 * same state. */
ccl_device void integrator_shade_shadow_neural_record(KernelGlobals kg,
                                                      IntegratorShadowState state,
                                                      const uint packed_num_hits,
                                                      const uint neural_slot)
{
  const Spectrum throughput = INTEGRATOR_STATE(state, shadow_path, throughput);
  const uint16_t transparent_bounce = INTEGRATOR_STATE(state, shadow_path, transparent_bounce);
  const uint16_t volume_bounds_bounce = INTEGRATOR_STATE(state, shadow_path, volume_bounds_bounce);
  const uint16_t rng_offset = INTEGRATOR_STATE(state, shadow_path, rng_offset);
  const float tmin = INTEGRATOR_STATE(state, shadow_ray, tmin);
  const int stack_size = min(int(kernel_data.volume_stack_size), MAX_VOLUME_STACK_SIZE);
  VolumeStack stack[MAX_VOLUME_STACK_SIZE];
  for (int i = 0; i < stack_size; i++) {
    stack[i] = integrator_state_read_shadow_volume_stack(state, i);
  }

  uint cursor = 0;
  integrate_transparent_shadow(
      kg, state, packed_num_hits, SRNXR_QUEUE_RECORD, neural_slot, &cursor, 0);

  INTEGRATOR_STATE_WRITE(state, shadow_path, throughput) = throughput;
  INTEGRATOR_STATE_WRITE(state, shadow_path, transparent_bounce) = transparent_bounce;
  INTEGRATOR_STATE_WRITE(state, shadow_path, volume_bounds_bounce) = volume_bounds_bounce;
  INTEGRATOR_STATE_WRITE(state, shadow_path, rng_offset) = rng_offset;
  INTEGRATOR_STATE_WRITE(state, shadow_path, packed_num_hits) = packed_num_hits;
  INTEGRATOR_STATE_WRITE(state, shadow_ray, tmin) = tmin;
  for (int i = 0; i < stack_size; i++) {
    integrator_state_write_shadow_volume_stack(state, i, stack[i]);
  }
  kernel_integrator_state.srnxr_queue.window_count[neural_slot] = cursor;
}
#endif

ccl_device void integrator_shade_shadow(KernelGlobals kg,
                                        IntegratorShadowState state,
                                        ccl_global float *ccl_restrict render_buffer,
                                        const uint neural_mode = 0,
                                        const uint neural_slot = 0)
{
  PROFILING_INIT(kg, PROFILING_SHADE_SHADOW_SETUP);
  const uint packed_num_hits = INTEGRATOR_STATE(state, shadow_path, packed_num_hits);

#ifdef __TRANSPARENT_SHADOWS__
  uint neural_count = 0;
#  if defined(WITH_CYCLES_NEURAL_VOLUME) && defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
  if (neural_mode == SRNXR_QUEUE_RECORD) {
    integrator_shade_shadow_neural_record(kg, state, packed_num_hits, neural_slot);
    return;
  }
  if (neural_mode == SRNXR_QUEUE_REPLAY) {
    neural_count = kernel_integrator_state.srnxr_queue.window_count[neural_slot];
  }
#  endif
  uint neural_cursor = 0;
  /* Evaluate transparent shadows. */
  const TransparentShadowEvalResult result = integrate_transparent_shadow(
      kg, state, packed_num_hits, neural_mode, neural_slot, &neural_cursor, neural_count);
  if (result == TRANSPARENT_SHADOW_EVAL_CACHE_MISS) {
    integrator_shadow_path_cache_miss(state, DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
    return;
  }
  if (result == TRANSPARENT_SHADOW_EVAL_OPAQUE) {
    integrator_shadow_path_terminate(state, DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
    return;
  }
#endif

  if (shadow_intersections_has_remaining(packed_num_hits)) {
    /* More intersections to find, continue shadow ray. */
    integrator_shadow_path_next(
        state, DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
    return;
  }

  guiding_record_direct_light(kg, state);
  film_write_direct_light(kg, state, render_buffer);
  integrator_shadow_path_terminate(state, DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW);
}

CCL_NAMESPACE_END
