/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Decoder of the SRNXR device version 2 blob (kernel/util/image_3d_srnxr.h).
 *
 * Shared by the Cycles kernels (CPU, CUDA, OptiX) and the standalone tools in tools/srnxr,
 * which provide the ccl_device* and ccl_global macros. The mathematical contract is
 * documented in tools/srnxr/neural_volume.md and implemented independently by the numpy oracle
 * tools/srnxr/srnxr_reference.py:
 *
 *   p = (P, time_frame + time_shutter * (ray_time - 0.5))
 *   outside the domain box             -> outside_value
 *   L = strict leaf of p               (octree descent, coordinates equal to the node center
 *                                       go to the upper child)
 *   for leaf j in neighbors(L), p in the closed expanded box of j:
 *     x = (p - input_offset_j) * input_scale_j
 *     h = sin(w0_l * (W_l h + b_l)), l = 0 .. layers - 1, h_0 = x
 *     u_j = (W_head h + b_head) * output_scale_j + output_offset_j
 *     e_j = -|(p - blend_center_j) * blend_scale_j|^2 / (2 sigma_j^2)
 *   u = sum exp(e_j - max e) u_j / sum exp(e_j - max e)
 *
 * The sum over j uses the running maximum (as in an online softmax), so the weight sum is
 * always >= 1 and never underflows. */

CCL_NAMESPACE_BEGIN

#if defined(__KERNEL_CUDA__) && !defined(__KERNEL_OPTIX__) && defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 800) && \
    (defined(SRNXR_STANDALONE) || defined(WITH_CYCLES_NEURAL_CUDA_TENSOR))
/* mma.sync.m16n8k16 with FP16 inputs and FP32 accumulation, warp intrinsics. Not in OptiX
 * programs, where warp-synchronous operations are not allowed. */
#  define SRNXR_HAS_MMA 1
#endif

#if defined(__KERNEL_OPTIX__) && defined(__CUDA_ARCH__) && defined(OPTIX_VERSION) && \
    (OPTIX_VERSION >= 90000) && defined(OPTIX_INCLUDE_COOPERATIVE_VECTOR) && \
    (defined(SRNXR_STANDALONE) || defined(WITH_CYCLES_NEURAL_COOPVEC))
/* OptiX cooperative vectors, only inside OptiX programs. */
#  if OPTIX_INCLUDE_COOPERATIVE_VECTOR
#    define SRNXR_HAS_COOPVEC 1
#  endif
#endif

#if defined(__KERNEL_CUDA__) && !defined(__KERNEL_OPTIX__) && defined(__CUDA_ARCH__)
#  define SRNXR_HAS_WARP_INTRINSICS 1
#  if defined(SRNXR_STANDALONE) || defined(WITH_CYCLES_NEURAL_CUDA_FUSED)
/* Warp-cooperative SIMT path of CUDA_FUSED */
#    define SRNXR_HAS_WARP_SIMT 1
#  endif
#endif

ccl_device_forceinline float srnxr_half_to_float(const uint16_t h)
{
#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
#else
  /* Exact for every half value. */
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  const uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits;
  if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  }
  else if (exponent != 0) {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  else if (mantissa == 0) {
    bits = sign;
  }
  else {
    int e = -1;
    do {
      e++;
      mantissa <<= 1;
    } while ((mantissa & 0x400u) == 0);
    bits = sign | (uint32_t(112 - e) << 23) | ((mantissa & 0x3FFu) << 13);
  }
  union {
    uint32_t u;
    float f;
  } v;
  v.u = bits;
  return v.f;
#endif
}

/* Cycles compiles its CUDA kernels with --use_fast_math, which turns sinf() and expf() into
 * the approximate intrinsics and divisions into approximate divisions. The decoder calls the
 * libdevice functions and the IEEE division directly, so the accurate mode stays accurate in
 * every build (the flush of denormals to zero of fast math still applies). */
#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_expf(float);
#endif

ccl_device_forceinline float srnxr_sin(const float x, const uint32_t sine_mode)
{
#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
  if (sine_mode == SRNXR_SINE_FAST) {
    return __sinf(x);
  }
  return __nv_sinf(x);
#else
  (void)sine_mode;
  return sinf(x);
#endif
}

ccl_device_forceinline float srnxr_exp(const float x)
{
#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
  return __nv_expf(x);
#else
  return expf(x);
#endif
}

ccl_device_forceinline float srnxr_div(const float a, const float b)
{
#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

ccl_device_forceinline const ccl_global SRNXRDeviceHeaderV2 *srnxr_v2_header(
    const ccl_global uint8_t *base)
{
  return reinterpret_cast<const ccl_global SRNXRDeviceHeaderV2 *>(base);
}

/* Statistics, aggregated per warp on CUDA. The standalone tools and OptiX programs keep them in
 * the blob, the Cycles CUDA kernels in the module global srnxr_counters
 * (kernel/device/cuda/globals.h, read back by CUDADevice). Not updated on the CPU. */
ccl_device_forceinline void srnxr_counter_add(const ccl_global uint8_t *base,
                                              const SRNXRCounter counter,
                                              const uint32_t n)
{
#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  if (!(hdr->flags & SRNXR_DEVICE_FLAG_COUNTERS)) {
    return;
  }
#  if defined(SRNXR_STANDALONE) || defined(__KERNEL_OPTIX__)
  /* In the blob, read back by the tools and by OptiXDevice when it frees the image. */
  unsigned long long int *counters = (unsigned long long int *)(base + hdr->counters_offset);
#  else
  unsigned long long int *counters = srnxr_counters +
                                     (hdr->counter_slot % SRNXR_COUNTER_SLOTS) * SRNXR_COUNTER_NUM;
#  endif
#  if defined(SRNXR_HAS_WARP_INTRINSICS) && (__CUDA_ARCH__ >= 800)
  /* Lanes may update different counters in the same call, aggregate per counter. */
  const uint32_t mask = __match_any_sync(__activemask(), uint32_t(counter));
  const uint32_t total = __reduce_add_sync(mask, n);
  uint32_t lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  if (lane == uint32_t(__ffs(mask) - 1) && total) {
    atomicAdd(counters + counter, (unsigned long long int)total);
  }
#  else
  if (n) {
    atomicAdd(counters + counter, (unsigned long long int)n);
  }
#  endif
#else
  (void)base;
  (void)counter;
  (void)n;
#endif
}

ccl_device_inline uint32_t srnxr_v2_find_leaf(const ccl_global uint8_t *base, const float p[4])
{
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNodeV2 *nodes = reinterpret_cast<const ccl_global SRNXRNodeV2 *>(
      base + hdr->nodes_offset);
  uint32_t n = 0;
  for (int depth = 0; depth <= 32; depth++) {
    const ccl_global SRNXRNodeV2 &node = nodes[n];
    if (node.first_child == SRNXR_INVALID_INDEX) {
      return node.leaf;
    }
    const float cx = 0.5f * (node.bmin[0] + node.bmax[0]);
    const float cy = 0.5f * (node.bmin[1] + node.bmax[1]);
    const float cz = 0.5f * (node.bmin[2] + node.bmax[2]);
    const uint32_t k = uint32_t(p[0] >= cx) | (uint32_t(p[1] >= cy) << 1) |
                       (uint32_t(p[2] >= cz) << 2);
    n = node.first_child + k;
  }
  return SRNXR_INVALID_INDEX;
}

ccl_device_forceinline bool srnxr_v2_in_expanded(const ccl_global SRNXRLeafV2 &leaf,
                                                 const float p[4])
{
  return p[0] >= leaf.expanded_min[0] && p[0] <= leaf.expanded_max[0] &&
         p[1] >= leaf.expanded_min[1] && p[1] <= leaf.expanded_max[1] &&
         p[2] >= leaf.expanded_min[2] && p[2] <= leaf.expanded_max[2] &&
         p[3] >= leaf.expanded_min[3] && p[3] <= leaf.expanded_max[3];
}

ccl_device_forceinline float srnxr_v2_blend_exponent(const ccl_global SRNXRLeafV2 &leaf,
                                                     const float p[4])
{
  float r2 = 0.0f;
  for (int d = 0; d < 4; d++) {
    const float v = (p[d] - leaf.blend_center[d]) * leaf.blend_scale[d];
    r2 += v * v;
  }
  return -r2 * leaf.blend_inv_two_sigma_sq;
}

/* Running max-shifted Gaussian blending. */
struct SRNXRBlend {
  float max_exponent;
  float weight_sum;
  float value[SRNXR_V2_MAX_OUTPUTS];
};

ccl_device_forceinline void srnxr_blend_init(ccl_private SRNXRBlend &b)
{
  b.max_exponent = -FLT_MAX;
  b.weight_sum = 0.0f;
  for (int k = 0; k < SRNXR_V2_MAX_OUTPUTS; k++) {
    b.value[k] = 0.0f;
  }
}

ccl_device_forceinline void srnxr_blend_add(ccl_private SRNXRBlend &b,
                                            const float e,
                                            const ccl_private float *u,
                                            const uint32_t output_dim)
{
  if (e > b.max_exponent) {
    const float s = (b.weight_sum > 0.0f) ? srnxr_exp(b.max_exponent - e) : 0.0f;
    b.weight_sum = b.weight_sum * s + 1.0f;
    for (uint32_t k = 0; k < output_dim; k++) {
      b.value[k] = b.value[k] * s + u[k];
    }
    b.max_exponent = e;
  }
  else {
    const float w = srnxr_exp(e - b.max_exponent);
    b.weight_sum += w;
    for (uint32_t k = 0; k < output_dim; k++) {
      b.value[k] += w * u[k];
    }
  }
}

ccl_device_forceinline void srnxr_v2_local_input(const ccl_global SRNXRLeafV2 &leaf,
                                                 const float p[4],
                                                 ccl_private float x[4])
{
  for (int d = 0; d < 4; d++) {
    x[d] = (p[d] - leaf.input_offset[d]) * leaf.input_scale[d];
  }
}

/* Dot product of a FP16 weight row with FP32 activations, FP32 accumulation in index order. */
ccl_device_forceinline float srnxr_dot_row(const ccl_global uint16_t *row,
                                           const ccl_private float *a,
                                           const uint32_t n,
                                           float sum)
{
#if defined(__KERNEL_GPU__)
  if ((n & 7) == 0) {
    /* Rows of widths divisible by 8 are 16-byte aligned (srnxr_model.cpp). */
    const ccl_global uint4 *row4 = reinterpret_cast<const ccl_global uint4 *>(row);
    for (uint32_t i = 0; i < n; i += 8) {
      const uint4 w = row4[i >> 3];
      const uint32_t words[4] = {w.x, w.y, w.z, w.w};
      for (int j = 0; j < 4; j++) {
        sum += srnxr_half_to_float(uint16_t(words[j] & 0xFFFFu)) * a[i + 2 * j];
        sum += srnxr_half_to_float(uint16_t(words[j] >> 16)) * a[i + 2 * j + 1];
      }
    }
    return sum;
  }
#endif
  for (uint32_t i = 0; i < n; i++) {
    sum += srnxr_half_to_float(row[i]) * a[i];
  }
  return sum;
}

/* SRNXR_SIMT_ROWS consecutive rows of a FP16 weight matrix (row length n) with the same FP32
 * activations: independent accumulators hide the FMA and load latency of a lone thread and every
 * activation is read once for all rows. Each row sums in index order like srnxr_dot_row. */
#define SRNXR_SIMT_ROWS 8
ccl_device_forceinline void srnxr_dot_rows(const ccl_global uint16_t *rows,
                                           const ccl_private float *a,
                                           const uint32_t n,
                                           const ccl_global float *bias,
                                           ccl_private float *sum)
{
  for (int r = 0; r < SRNXR_SIMT_ROWS; r++) {
    sum[r] = bias[r];
  }
#if defined(__KERNEL_GPU__)
  if ((n & 7) == 0) {
    const ccl_global uint4 *row4 = reinterpret_cast<const ccl_global uint4 *>(rows);
    const uint32_t n8 = n >> 3;
    for (uint32_t i = 0; i < n8; i++) {
      uint32_t words[SRNXR_SIMT_ROWS][4];
      for (int r = 0; r < SRNXR_SIMT_ROWS; r++) {
        const uint4 w = row4[r * n8 + i];
        words[r][0] = w.x;
        words[r][1] = w.y;
        words[r][2] = w.z;
        words[r][3] = w.w;
      }
      for (int j = 0; j < 4; j++) {
        const float a0 = a[8 * i + 2 * j];
        const float a1 = a[8 * i + 2 * j + 1];
        for (int r = 0; r < SRNXR_SIMT_ROWS; r++) {
          sum[r] += srnxr_half_to_float(uint16_t(words[r][j] & 0xFFFFu)) * a0;
          sum[r] += srnxr_half_to_float(uint16_t(words[r][j] >> 16)) * a1;
        }
      }
    }
    return;
  }
#endif
  for (int r = 0; r < SRNXR_SIMT_ROWS; r++) {
    sum[r] = srnxr_dot_row(rows + r * n, a, n, sum[r]);
  }
}

/* One local SIREN, one query per thread: FP16 weights, FP32 arithmetic and sine. */
ccl_device_noinline void srnxr_v2_eval_network_simt(const ccl_global uint8_t *base,
                                                    const uint32_t network,
                                                    const ccl_private float *x,
                                                    ccl_private float *y)
{
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      base + hdr->networks_offset)[network];
  const ccl_global uint16_t *weights = reinterpret_cast<const ccl_global uint16_t *>(
                                           base + hdr->weights_offset) +
                                       net.weight_offset;
  const ccl_global float *biases = reinterpret_cast<const ccl_global float *>(
                                       base + hdr->biases_offset) +
                                   net.bias_offset;
  const uint32_t width = net.width;
  const uint32_t in_dim = hdr->input_dim;
  const uint32_t sine_mode = hdr->sine_mode;

  float a[SRNXR_V2_MAX_WIDTH];
  float b[SRNXR_V2_MAX_WIDTH];

  float w0 = hdr->layer_w0[0];
  for (uint32_t o = 0; o < width; o++) {
    const ccl_global uint16_t *row = weights + o * in_dim;
    float sum = biases[o];
    for (uint32_t d = 0; d < in_dim; d++) {
      sum += srnxr_half_to_float(row[d]) * x[d];
    }
    a[o] = srnxr_sin(w0 * sum, sine_mode);
  }
  weights += width * in_dim;
  biases += width;

  for (uint32_t layer = 1; layer < hdr->hidden_layer_count; layer++) {
    w0 = hdr->layer_w0[layer];
    uint32_t o = 0;
    for (; o + SRNXR_SIMT_ROWS <= width; o += SRNXR_SIMT_ROWS) {
      float sum[SRNXR_SIMT_ROWS];
      srnxr_dot_rows(weights + o * width, a, width, biases + o, sum);
      for (int r = 0; r < SRNXR_SIMT_ROWS; r++) {
        b[o + r] = srnxr_sin(w0 * sum[r], sine_mode);
      }
    }
    for (; o < width; o++) {
      b[o] = srnxr_sin(w0 * srnxr_dot_row(weights + o * width, a, width, biases[o]), sine_mode);
    }
    for (o = 0; o < width; o++) {
      a[o] = b[o];
    }
    weights += width * width;
    biases += width;
  }

  for (uint32_t k = 0; k < hdr->output_dim; k++) {
    y[k] = srnxr_dot_row(weights + k * width, a, width, biases[k]);
  }
}

ccl_device_forceinline void srnxr_v2_leaf_output(const ccl_global SRNXRLeafV2 &leaf,
                                                 ccl_private float *y,
                                                 const uint32_t output_dim)
{
  for (uint32_t k = 0; k < output_dim; k++) {
    y[k] = y[k] * leaf.output_scale[k] + leaf.output_offset[k];
  }
}

#ifdef SRNXR_HAS_COOPVEC
// ----------------------------------------------------------------------------
// COOPVEC: OptiX cooperative vectors (one query per thread inside an OptiX program)
// ----------------------------------------------------------------------------
// The first layer (4 inputs) is evaluated in FP32 like the other backends, its activations
// are rounded to FP16 into the input vector of the hidden layer products. The hidden and
// output layers are optixCoopVecMatMul() with FP16 matrices in the INFERENCING_OPTIMAL
// layout, FP16 input vectors and FP16 output vectors and biases (coopvec_mode 1): OptiX 9.1
// rejects FP32 output vectors for FP16 inputs at module compile time ("output vector type is
// of unsupported type"), so every layer output is rounded to FP16 before the sine. The FP32
// output variant (coopvec_mode 0) is only compiled with SRNXR_COOPVEC_TRY_FP32_OUTPUT, to
// reproduce that error. OptiX has no cooperative vector sine: the sine is applied per element
// in FP32 (srnxr_sin), which takes the vector apart.

ccl_device_forceinline uint32_t srnxr_align64(const uint32_t v)
{
  return (v + 63u) & ~63u;
}

/* Called per vector element: inlining the accurate sine (with its large argument path) into the
 * fully unrolled element loops makes the OptiX module too large to compile in practice. */
ccl_device_noinline float srnxr_coopvec_sin(const float x, const uint32_t sine_mode)
{
  return srnxr_sin(x, sine_mode);
}

template<int W>
ccl_device_noinline void srnxr_v2_eval_network_coopvec(const ccl_global uint8_t *base,
                                                       const uint32_t network,
                                                       const ccl_private float *x,
                                                       ccl_private float *y)
{
  using VH = OptixCoopVec<__half, W>;
  using VF = OptixCoopVec<float, W>;
  using VF8 = OptixCoopVec<float, 8>;
  using VH8 = OptixCoopVec<__half, 8>;
  constexpr OptixCoopVecElemType F16 = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
  constexpr OptixCoopVecElemType F32 = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32;
  constexpr OptixCoopVecMatrixLayout LAYOUT = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;

  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      base + hdr->networks_offset)[network];
  const ccl_global uint16_t *w0_weights = reinterpret_cast<const ccl_global uint16_t *>(
                                              base + hdr->weights_offset) +
                                          net.weight_offset;
  const ccl_global float *b0 = reinterpret_cast<const ccl_global float *>(base +
                                                                         hdr->biases_offset) +
                                net.bias_offset;
  const CUdeviceptr region = CUdeviceptr(base + hdr->coopvec_offset);
  const uint32_t layers = hdr->hidden_layer_count;
  const uint32_t in_dim = hdr->input_dim;
  const uint32_t sine_mode = hdr->sine_mode;
  const uint32_t size_ww = optixCoopVecGetMatrixSize<W, W, F16, LAYOUT>();
  const uint32_t size_head = optixCoopVecGetMatrixSize<8, W, F16, LAYOUT>();
  uint32_t matrix = net.coopvec_offset;
  const uint32_t bias32 = matrix + (layers - 1) * size_ww + size_head;
  const uint32_t bias16 = bias32 + srnxr_align64(((layers - 1) * W + 8) * 4);

  VH a;
  {
    const float w0 = hdr->layer_w0[0];
#  pragma unroll 1
    for (int o = 0; o < W; o++) {
      const ccl_global uint16_t *row = w0_weights + o * in_dim;
      float sum = b0[o];
      for (uint32_t d = 0; d < in_dim; d++) {
        sum += srnxr_half_to_float(row[d]) * x[d];
      }
      a[o] = __float2half_rn(srnxr_coopvec_sin(w0 * sum, sine_mode));
    }
  }

  for (uint32_t layer = 1; layer < layers; layer++) {
    const float w0 = hdr->layer_w0[layer];
#  ifdef SRNXR_COOPVEC_TRY_FP32_OUTPUT
    if (hdr->coopvec_mode == 0) {
      const VF z = optixCoopVecMatMul<VF, VH, F16, LAYOUT, false, W, W, F16, F32>(
          a, region, matrix, region, bias32 + (layer - 1) * W * 4);
#  pragma unroll 1
      for (int i = 0; i < W; i++) {
        a[i] = __float2half_rn(srnxr_coopvec_sin(w0 * z[i], sine_mode));
      }
    }
    else
#  endif
    {
      const VH z = optixCoopVecMatMul<VH, VH, F16, LAYOUT, false, W, W, F16, F16>(
          a, region, matrix, region, bias16 + (layer - 1) * W * 2);
#  pragma unroll 1
      for (int i = 0; i < W; i++) {
        a[i] = __float2half_rn(srnxr_coopvec_sin(w0 * __half2float(z[i]), sine_mode));
      }
    }
    matrix += size_ww;
  }

#  ifdef SRNXR_COOPVEC_TRY_FP32_OUTPUT
  if (hdr->coopvec_mode == 0) {
    const VF8 out = optixCoopVecMatMul<VF8, VH, F16, LAYOUT, false, 8, W, F16, F32>(
        a, region, matrix, region, bias32 + (layers - 1) * W * 4);
    for (uint32_t k = 0; k < hdr->output_dim; k++) {
      y[k] = out[k];
    }
  }
  else
#  endif
  {
    const VH8 out = optixCoopVecMatMul<VH8, VH, F16, LAYOUT, false, 8, W, F16, F16>(
        a, region, matrix, region, bias16 + (layers - 1) * W * 2);
    for (uint32_t k = 0; k < hdr->output_dim; k++) {
      y[k] = __half2float(out[k]);
    }
  }
}
#endif /* SRNXR_HAS_COOPVEC */

#ifdef SRNXR_COOPVEC_DIRECT_CALLABLES
/* In Cycles the cooperative vector evaluation is compiled once, as the direct callables 0, 1, 2
 * (widths 64, 128, 256) of kernel_neural_volume.cu: inlined into every call site of the volume
 * shading programs it makes the OptiX module too large to compile in practice. Inputs and
 * outputs are passed by value. */
struct SRNXRCoopVecOut {
  float v[SRNXR_V2_MAX_OUTPUTS];
};

template<int W>
ccl_device_forceinline SRNXRCoopVecOut srnxr_coopvec_callable_body(const ccl_global uint8_t *base,
                                                                   const uint32_t network,
                                                                   const float x0,
                                                                   const float x1,
                                                                   const float x2,
                                                                   const float x3)
{
  const float x[4] = {x0, x1, x2, x3};
  SRNXRCoopVecOut out;
  srnxr_v2_eval_network_coopvec<W>(base, network, x, out.v);
  return out;
}
#endif

/* SIMT decode of the neighbor positions [begin, end) of the strict leaf. */
ccl_device_inline void srnxr_v2_decode_simt_range(const ccl_global uint8_t *base,
                                                  const float p[4],
                                                  const uint32_t begin,
                                                  const uint32_t end,
                                                  ccl_private SRNXRBlend &blend,
                                                  const SRNXRCounter counter)
{
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRLeafV2 *leaves = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
      base + hdr->leaves_offset);
  const ccl_global uint32_t *neighbors = reinterpret_cast<const ccl_global uint32_t *>(
      base + hdr->neighbors_offset);
  uint32_t evals = 0, coopvec_evals = 0;
  for (uint32_t i = begin; i < end; i++) {
    const ccl_global SRNXRLeafV2 &leaf = leaves[neighbors[i]];
    if (!srnxr_v2_in_expanded(leaf, p)) {
      continue;
    }
    float x[4], y[SRNXR_V2_MAX_OUTPUTS];
    srnxr_v2_local_input(leaf, p, x);
#ifdef SRNXR_HAS_COOPVEC
    if (hdr->backend == SRNXR_BACKEND_COOPVEC && hdr->coopvec_offset) {
      const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
          base + hdr->networks_offset)[leaf.network];
      bool done = false;
#  ifdef SRNXR_COOPVEC_DIRECT_CALLABLES
      if (net.coopvec_offset != SRNXR_INVALID_INDEX &&
          (net.width == 64 || net.width == 128 || net.width == 256))
      {
        const unsigned int index = (net.width == 64) ? 0u : (net.width == 128) ? 1u : 2u;
        const SRNXRCoopVecOut out =
            optixDirectCall<SRNXRCoopVecOut, const ccl_global uint8_t *, uint32_t, float, float,
                            float, float>(index, base, leaf.network, x[0], x[1], x[2], x[3]);
        for (uint32_t k = 0; k < hdr->output_dim; k++) {
          y[k] = out.v[k];
        }
        done = true;
      }
#  else
      if (net.coopvec_offset != SRNXR_INVALID_INDEX) {
        if (net.width == 256) {
          srnxr_v2_eval_network_coopvec<256>(base, leaf.network, x, y);
          done = true;
        }
        else if (net.width == 128) {
          srnxr_v2_eval_network_coopvec<128>(base, leaf.network, x, y);
          done = true;
        }
        else if (net.width == 64) {
          srnxr_v2_eval_network_coopvec<64>(base, leaf.network, x, y);
          done = true;
        }
      }
#  endif
      if (done) {
        coopvec_evals++;
      }
      else {
        srnxr_v2_eval_network_simt(base, leaf.network, x, y);
        evals++;
      }
    }
    else
#endif
    {
      srnxr_v2_eval_network_simt(base, leaf.network, x, y);
      evals++;
    }
    srnxr_v2_leaf_output(leaf, y, hdr->output_dim);
    srnxr_blend_add(blend, srnxr_v2_blend_exponent(leaf, p), y, hdr->output_dim);
  }
  srnxr_counter_add(base, SRNXR_COUNTER_SIMT_EVALS, evals);
  if (counter != SRNXR_COUNTER_SIMT_EVALS) {
    srnxr_counter_add(base, counter, evals);
  }
  srnxr_counter_add(base, SRNXR_COUNTER_COOPVEC_EVALS, coopvec_evals);
  srnxr_counter_add(base, SRNXR_COUNTER_NETWORK_EVALS, evals + coopvec_evals);
}

// ----------------------------------------------------------------------------
// Warp-cooperative evaluation (CUDA, not in OptiX programs)
// ----------------------------------------------------------------------------
//
// When all 32 lanes of a warp reach the decoder together, the queries that need the same
// network are evaluated together. srnxr_v2_decode_warp() repeatedly lets the lowest lane with
// pending work pick the network of its next candidate leaf; every lane with a pending
// candidate of that network joins as one row. Lanes without a query (outside the domain,
// beyond the batch) take part with an empty candidate list.
//
// CUDA_FUSED, warp-cooperative SIMT (srnxr_v2_eval_network_warp): lane l owns the neurons
// 64 j + 2 l + {0, 1} of every layer for G rows at a time, the inputs of a layer are broadcast
// with shuffles and the FP16 weights are read from a transposed [in][out] copy, coalesced.
// FP32 multiply-accumulate in the same order as the per-thread path (bias, then the inputs in
// index order); only the output layer is summed as a warp reduction.
//
// CUDA_TENSOR (srnxr_v2_eval_network_tensor): the rows form a 16 x K activation tile A
// (FP16), the weights are the B operand (FP16, packed by srnxr_model.cpp in fragment order),
// mma.sync.m16n8k16 accumulates in FP32. Lane l = 4 g + t holds rows g and g + 8 and columns
// 2t, 2t + 1, 2t + 8, 2t + 9 of every 16 column block, which is also the layout of the FP32
// accumulator of two adjacent 8 column output tiles: a layer output becomes the A operand of
// the next layer without a shuffle. Bias, w0 and sine are applied in FP32 to the accumulators,
// the activations are rounded to FP16 once per layer. The first layer (4 inputs) is evaluated
// in FP32 from the FP16 weights, as the SIMT path does.

#ifdef SRNXR_HAS_WARP_INTRINSICS

ccl_device_forceinline uint32_t srnxr_lane_id()
{
  uint32_t lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  return lane;
}

/* Lane of the n-th (0-based) set bit of mask, 32 if there is none. */
ccl_device_forceinline uint32_t srnxr_nth_lane(const uint32_t mask, const uint32_t n)
{
  const uint32_t lane = __fns(mask, 0, int(n) + 1);
  return (lane == 0xFFFFFFFFu) ? 32u : lane;
}

ccl_device_forceinline float srnxr_half_lo(const uint32_t w)
{
  return srnxr_half_to_float(uint16_t(w & 0xFFFFu));
}

ccl_device_forceinline float srnxr_half_hi(const uint32_t w)
{
  return srnxr_half_to_float(uint16_t(w >> 16));
}

template<int W, int G>
ccl_device_noinline void srnxr_v2_eval_network_warp(const ccl_global uint8_t *base,
                                                    const uint32_t network,
                                                    const uint32_t rows,
                                                    const ccl_private float *x,
                                                    ccl_private float *y)
{
  constexpr int J = W / 64;
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      base + hdr->networks_offset)[network];
  const ccl_global uint16_t *weights = reinterpret_cast<const ccl_global uint16_t *>(
                                           base + hdr->weights_offset) +
                                       net.weight_offset;
  const ccl_global float *biases = reinterpret_cast<const ccl_global float *>(
                                       base + hdr->biases_offset) +
                                   net.bias_offset;
  const ccl_global uint32_t *wt_base = reinterpret_cast<const ccl_global uint32_t *>(
      reinterpret_cast<const ccl_global uint16_t *>(base + hdr->warp_weights_offset) +
      net.warp_offset);
  const uint32_t in_dim = hdr->input_dim;
  const uint32_t output_dim = hdr->output_dim;
  const uint32_t layers = hdr->hidden_layer_count;
  const uint32_t sine_mode = hdr->sine_mode;
  const uint32_t lane = srnxr_lane_id();
  const uint32_t nrows = __popc(rows);
  const uint32_t rank = __popc(rows & ((1u << lane) - 1u));
  const ccl_global uint16_t *head = weights + W * in_dim + (layers - 1) * W * W;
  const ccl_global float *head_bias = biases + layers * W;

  for (uint32_t r0 = 0; r0 < nrows; r0 += G) {
    float xs[G][4];
#  pragma unroll
    for (int s = 0; s < G; s++) {
      const uint32_t src = srnxr_nth_lane(rows, r0 + s);
#  pragma unroll
      for (int d = 0; d < 4; d++) {
        const float v = __shfl_sync(0xFFFFFFFFu, x[d], src & 31);
        xs[s][d] = (src < 32) ? v : 0.0f;
      }
    }

    /* First layer */
    float act[G][2 * J];
    {
      const float w0 = hdr->layer_w0[0];
#  pragma unroll
      for (int n = 0; n < 2 * J; n++) {
        const uint32_t o = 64 * (n >> 1) + 2 * lane + (n & 1);
        const ccl_global uint16_t *row = weights + o * in_dim;
        float wv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t d = 0; d < in_dim; d++) {
          wv[d] = srnxr_half_to_float(row[d]);
        }
        const float b = biases[o];
#  pragma unroll
        for (int s = 0; s < G; s++) {
          float sum = b;
          for (uint32_t d = 0; d < in_dim; d++) {
            sum += wv[d] * xs[s][d];
          }
          act[s][n] = srnxr_sin(w0 * sum, sine_mode);
        }
      }
    }

    /* Hidden layers */
    const ccl_global uint32_t *wt = wt_base;
    const ccl_global float *bl = biases + W;
    for (uint32_t layer = 1; layer < layers; layer++) {
      float acc[G][2 * J];
#  pragma unroll
      for (int n = 0; n < 2 * J; n++) {
        const float b = bl[64 * (n >> 1) + 2 * lane + (n & 1)];
#  pragma unroll
        for (int s = 0; s < G; s++) {
          acc[s][n] = b;
        }
      }
#  pragma unroll
      for (int jj = 0; jj < J; jj++) {
        for (uint32_t src = 0; src < 32; src++) {
#  pragma unroll
          for (int e = 0; e < 2; e++) {
            const uint32_t i = 64 * jj + 2 * src + e;
            uint32_t wq[J];
#  pragma unroll
            for (int j = 0; j < J; j++) {
              wq[j] = wt[i * (W / 2) + lane + 32 * j];
            }
#  pragma unroll
            for (int s = 0; s < G; s++) {
              const float v = __shfl_sync(0xFFFFFFFFu, act[s][2 * jj + e], src);
#  pragma unroll
              for (int j = 0; j < J; j++) {
                acc[s][2 * j] += srnxr_half_lo(wq[j]) * v;
                acc[s][2 * j + 1] += srnxr_half_hi(wq[j]) * v;
              }
            }
          }
        }
      }
      const float w0 = hdr->layer_w0[layer];
#  pragma unroll
      for (int s = 0; s < G; s++) {
#  pragma unroll
        for (int n = 0; n < 2 * J; n++) {
          act[s][n] = srnxr_sin(w0 * acc[s][n], sine_mode);
        }
      }
      wt += W * W / 2;
      bl += W;
    }

    /* Output layer, row-major head weights, warp reduction. */
    float out[G];
    for (uint32_t k = 0; k < output_dim; k++) {
      float wk[2 * J];
#  pragma unroll
      for (int n = 0; n < 2 * J; n++) {
        wk[n] = srnxr_half_to_float(head[k * W + 64 * (n >> 1) + 2 * lane + (n & 1)]);
      }
#  pragma unroll
      for (int s = 0; s < G; s++) {
        float partial = 0.0f;
#  pragma unroll
        for (int n = 0; n < 2 * J; n++) {
          partial += wk[n] * act[s][n];
        }
#  pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          partial += __shfl_xor_sync(0xFFFFFFFFu, partial, offset);
        }
        out[s] = partial + head_bias[k];
      }
#  pragma unroll
      for (int s = 0; s < G; s++) {
        if (rank == r0 + s) {
          y[k] = out[s];
        }
      }
    }
  }
}

#endif /* SRNXR_HAS_WARP_INTRINSICS */

#ifdef SRNXR_HAS_MMA

ccl_device_forceinline uint32_t srnxr_pack_half2(const float lo, const float hi)
{
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

ccl_device_forceinline void srnxr_mma(float c[4], const uint32_t a[4], const uint2 b)
{
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}

/* Evaluate one network of width W for up to 16 rows. rows: lanes owning the rows in lane
 * order. Every lane of the warp must call this with the same network and rows. A lane that
 * owns a row passes its local input x and receives the network output y. */
template<int W>
ccl_device_noinline void srnxr_v2_eval_network_tensor(const ccl_global uint8_t *base,
                                                      const uint32_t network,
                                                      const uint32_t rows,
                                                      const ccl_private float *x,
                                                      ccl_private float *y)
{
  constexpr int KT = W / 16;
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      base + hdr->networks_offset)[network];
  const ccl_global uint16_t *w0_weights = reinterpret_cast<const ccl_global uint16_t *>(
                                              base + hdr->weights_offset) +
                                          net.weight_offset;
  const ccl_global float *biases = reinterpret_cast<const ccl_global float *>(
                                       base + hdr->biases_offset) +
                                   net.bias_offset;
  const ccl_global uint2 *mma = reinterpret_cast<const ccl_global uint2 *>(
                                    base + hdr->mma_weights_offset) +
                                (net.mma_offset >> 1);
  const uint32_t in_dim = hdr->input_dim;
  const uint32_t sine_mode = hdr->sine_mode;

  const uint32_t lane = srnxr_lane_id();
  const uint32_t g = lane >> 2, t = lane & 3;

  /* Inputs of rows g and g + 8. */
  const uint32_t src0 = srnxr_nth_lane(rows, g);
  const uint32_t src1 = srnxr_nth_lane(rows, g + 8);
  float x0[4], x1[4];
  for (int d = 0; d < 4; d++) {
    const float v0 = __shfl_sync(0xFFFFFFFFu, x[d], src0 & 31);
    const float v1 = __shfl_sync(0xFFFFFFFFu, x[d], src1 & 31);
    x0[d] = (src0 < 32) ? v0 : 0.0f;
    x1[d] = (src1 < 32) ? v1 : 0.0f;
  }

  /* First layer in FP32, written as the A fragments of the second layer. */
  uint32_t a[KT][4];
  {
    const float w0 = hdr->layer_w0[0];
#  pragma unroll
    for (int kt = 0; kt < KT; kt++) {
      float act0[4], act1[4];
#  pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t col = kt * 16 + 2 * t + (j & 1) + ((j >> 1) << 3);
        const ccl_global uint16_t *row = w0_weights + col * in_dim;
        float s0 = biases[col], s1 = biases[col];
        for (uint32_t d = 0; d < in_dim; d++) {
          const float wv = srnxr_half_to_float(row[d]);
          s0 += wv * x0[d];
          s1 += wv * x1[d];
        }
        act0[j] = srnxr_sin(w0 * s0, sine_mode);
        act1[j] = srnxr_sin(w0 * s1, sine_mode);
      }
      a[kt][0] = srnxr_pack_half2(act0[0], act0[1]);
      a[kt][1] = srnxr_pack_half2(act1[0], act1[1]);
      a[kt][2] = srnxr_pack_half2(act0[2], act0[3]);
      a[kt][3] = srnxr_pack_half2(act1[2], act1[3]);
    }
  }
  biases += W;

  /* Hidden layers. The outputs go to a local array (written and read once per layer), which
   * leaves the registers to the B fragments of a whole output tile pair: they are all loaded
   * before the first mma so the L2 latency is paid once per pair, not once per mma. */
  for (uint32_t layer = 1; layer < hdr->hidden_layer_count; layer++) {
    const float w0 = hdr->layer_w0[layer];
    uint32_t next[KT][4];
#  pragma unroll 1
    for (int ot = 0; ot < KT; ot++) {
      const ccl_global uint2 *b0 = mma + ((2 * ot) * KT) * 32 + lane;
      const ccl_global uint2 *b1 = mma + ((2 * ot + 1) * KT) * 32 + lane;
      uint2 bf0[KT], bf1[KT];
#  pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        bf0[kt] = b0[kt * 32];
        bf1[kt] = b1[kt * 32];
      }
      float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#  pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        srnxr_mma(c0, a[kt], bf0[kt]);
        srnxr_mma(c1, a[kt], bf1[kt]);
      }
      const uint32_t col0 = ot * 16 + 2 * t;
      const float bias0 = biases[col0], bias1 = biases[col0 + 1];
      const float bias8 = biases[col0 + 8], bias9 = biases[col0 + 9];
      next[ot][0] = srnxr_pack_half2(srnxr_sin(w0 * (c0[0] + bias0), sine_mode),
                                     srnxr_sin(w0 * (c0[1] + bias1), sine_mode));
      next[ot][1] = srnxr_pack_half2(srnxr_sin(w0 * (c0[2] + bias0), sine_mode),
                                     srnxr_sin(w0 * (c0[3] + bias1), sine_mode));
      next[ot][2] = srnxr_pack_half2(srnxr_sin(w0 * (c1[0] + bias8), sine_mode),
                                     srnxr_sin(w0 * (c1[1] + bias9), sine_mode));
      next[ot][3] = srnxr_pack_half2(srnxr_sin(w0 * (c1[2] + bias8), sine_mode),
                                     srnxr_sin(w0 * (c1[3] + bias9), sine_mode));
    }
#  pragma unroll
    for (int kt = 0; kt < KT; kt++) {
#  pragma unroll
      for (int j = 0; j < 4; j++) {
        a[kt][j] = next[kt][j];
      }
    }
    mma += (W / 8) * KT * 32;
    biases += W;
  }

  /* Linear head, one 8 column output tile. */
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  {
    uint2 bf[KT];
#  pragma unroll
    for (int kt = 0; kt < KT; kt++) {
      bf[kt] = mma[kt * 32 + lane];
    }
#  pragma unroll
    for (int kt = 0; kt < KT; kt++) {
      srnxr_mma(c, a[kt], bf[kt]);
    }
  }
  const uint32_t output_dim = hdr->output_dim;
  const uint32_t k0 = 2 * t;
  const float hb0 = (k0 < output_dim) ? biases[k0] : 0.0f;
  const float hb1 = (k0 + 1 < output_dim) ? biases[k0 + 1] : 0.0f;
  c[0] += hb0;
  c[1] += hb1;
  c[2] += hb0;
  c[3] += hb1;

  /* Row r of the output tile is in lanes 4 (r & 7) + k / 2, element (r < 8 ? 0 : 2) + k % 2. */
  const uint32_t rank = __popc(rows & ((1u << lane) - 1u));
  const bool upper = rank >= 8;
  for (uint32_t kp = 0; kp < (output_dim + 1) / 2; kp++) {
    const uint32_t src = (rank & 7) * 4 + kp;
    const float v0 = __shfl_sync(0xFFFFFFFFu, c[0], src);
    const float v1 = __shfl_sync(0xFFFFFFFFu, c[1], src);
    const float v2 = __shfl_sync(0xFFFFFFFFu, c[2], src);
    const float v3 = __shfl_sync(0xFFFFFFFFu, c[3], src);
    if (2 * kp < output_dim) {
      y[2 * kp] = upper ? v2 : v0;
    }
    if (2 * kp + 1 < output_dim) {
      y[2 * kp + 1] = upper ? v3 : v1;
    }
  }
}

// ----------------------------------------------------------------------------
// Block-cooperative Tensor Core evaluation: the warps of a block evaluate tiles of one network
// and share its weight fragments through shared memory (kernel/util/image_3d_srnxr_queue.h).
// ----------------------------------------------------------------------------
// The mma fragment copy of a hidden layer stores the two 8 column output tiles of a 16 column
// output tile pair contiguously (2 x KT x 32 uint2 = 8 KB for W = 256), which is the unit staged
// into shared memory by cp.async, double buffered. Every warp then reads the fragments of its
// own rows from shared memory, so the layer weights come from L2 once per block instead of once
// per warp.

/* Shared memory of one block: two staging buffers of the largest output tile pair. */
#  define SRNXR_TENSOR_BLOCK_STAGE_BYTES (2 * (256 / 16) * 32 * 8)
#  define SRNXR_TENSOR_BLOCK_SMEM_BYTES (2 * SRNXR_TENSOR_BLOCK_STAGE_BYTES)

ccl_device_forceinline void srnxr_cp_async16(void *smem, const ccl_global void *gmem)
{
  const uint32_t s = uint32_t(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

ccl_device_forceinline void srnxr_cp_async_commit()
{
  asm volatile("cp.async.commit_group;\n" ::);
}

template<int N> ccl_device_forceinline void srnxr_cp_async_wait()
{
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/* Stage bytes (a multiple of 16) from gmem to smem with every thread of the block. */
ccl_device_forceinline void srnxr_block_stage(ccl_private uint8_t *smem,
                                              const ccl_global uint8_t *gmem,
                                              const uint32_t bytes)
{
  const uint32_t chunks = bytes >> 4;
  for (uint32_t c = threadIdx.x; c < chunks; c += blockDim.x) {
    srnxr_cp_async16(smem + (c << 4), gmem + (c << 4));
  }
  srnxr_cp_async_commit();
}

/* Evaluate one network of width W for up to 16 rows per warp, all warps of the block with the
 * same network (a warp without rows passes rows = 0 and takes part in the staging). smem:
 * SRNXR_TENSOR_BLOCK_SMEM_BYTES of shared memory, 16 byte aligned. Every thread of the block
 * must call this (block-wide barriers). */
template<int W>
ccl_device_noinline void srnxr_v2_eval_network_tensor_block(const ccl_global uint8_t *base,
                                                            const uint32_t network,
                                                            const uint32_t rows,
                                                            const ccl_private float *x,
                                                            ccl_private float *y,
                                                            uint8_t *smem)
{
  constexpr int KT = W / 16;
  constexpr uint32_t pair_bytes = 2 * KT * 32 * 8;
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      base + hdr->networks_offset)[network];
  const ccl_global uint16_t *w0_weights = reinterpret_cast<const ccl_global uint16_t *>(
                                              base + hdr->weights_offset) +
                                          net.weight_offset;
  const ccl_global float *biases = reinterpret_cast<const ccl_global float *>(
                                       base + hdr->biases_offset) +
                                   net.bias_offset;
  const ccl_global uint2 *mma = reinterpret_cast<const ccl_global uint2 *>(
                                    base + hdr->mma_weights_offset) +
                                (net.mma_offset >> 1);
  const uint32_t in_dim = hdr->input_dim;
  const uint32_t sine_mode = hdr->sine_mode;
  const bool active = rows != 0;

  const uint32_t lane = srnxr_lane_id();
  const uint32_t g = lane >> 2, t = lane & 3;

  /* Inputs of rows g and g + 8. */
  const uint32_t src0 = srnxr_nth_lane(rows, g);
  const uint32_t src1 = srnxr_nth_lane(rows, g + 8);
  float x0[4], x1[4];
  for (int d = 0; d < 4; d++) {
    const float v0 = __shfl_sync(0xFFFFFFFFu, x[d], src0 & 31);
    const float v1 = __shfl_sync(0xFFFFFFFFu, x[d], src1 & 31);
    x0[d] = (src0 < 32) ? v0 : 0.0f;
    x1[d] = (src1 < 32) ? v1 : 0.0f;
  }

  /* First layer in FP32, written as the A fragments of the second layer. */
  uint32_t a[KT][4];
  if (active) {
    const float w0 = hdr->layer_w0[0];
#  pragma unroll
    for (int kt = 0; kt < KT; kt++) {
      float act0[4], act1[4];
#  pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t col = kt * 16 + 2 * t + (j & 1) + ((j >> 1) << 3);
        const ccl_global uint16_t *row = w0_weights + col * in_dim;
        float s0 = biases[col], s1 = biases[col];
        for (uint32_t d = 0; d < in_dim; d++) {
          const float wv = srnxr_half_to_float(row[d]);
          s0 += wv * x0[d];
          s1 += wv * x1[d];
        }
        act0[j] = srnxr_sin(w0 * s0, sine_mode);
        act1[j] = srnxr_sin(w0 * s1, sine_mode);
      }
      a[kt][0] = srnxr_pack_half2(act0[0], act0[1]);
      a[kt][1] = srnxr_pack_half2(act1[0], act1[1]);
      a[kt][2] = srnxr_pack_half2(act0[2], act0[3]);
      a[kt][3] = srnxr_pack_half2(act1[2], act1[3]);
    }
  }
  biases += W;

  /* Hidden layers: output tile pairs staged through shared memory, double buffered. */
  uint8_t *stage0 = smem;
  uint8_t *stage1 = smem + SRNXR_TENSOR_BLOCK_STAGE_BYTES;
  for (uint32_t layer = 1; layer < hdr->hidden_layer_count; layer++) {
    const float w0 = hdr->layer_w0[layer];
    uint32_t next[KT][4];
    srnxr_block_stage(stage0, reinterpret_cast<const ccl_global uint8_t *>(mma), pair_bytes);
#  pragma unroll 1
    for (int ot = 0; ot < KT; ot++) {
      uint8_t *cur = (ot & 1) ? stage1 : stage0;
      if (ot + 1 < KT) {
        uint8_t *nxt = (ot & 1) ? stage0 : stage1;
        srnxr_block_stage(nxt,
                          reinterpret_cast<const ccl_global uint8_t *>(mma + (2 * (ot + 1) * KT) * 32),
                          pair_bytes);
        srnxr_cp_async_wait<1>();
      }
      else {
        srnxr_cp_async_wait<0>();
      }
      __syncthreads();
      if (active) {
        const uint2 *b0 = reinterpret_cast<const uint2 *>(cur) + lane;
        const uint2 *b1 = b0 + KT * 32;
        float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#  pragma unroll
        for (int kt = 0; kt < KT; kt++) {
          srnxr_mma(c0, a[kt], b0[kt * 32]);
          srnxr_mma(c1, a[kt], b1[kt * 32]);
        }
        const uint32_t col0 = ot * 16 + 2 * t;
        const float bias0 = biases[col0], bias1 = biases[col0 + 1];
        const float bias8 = biases[col0 + 8], bias9 = biases[col0 + 9];
        next[ot][0] = srnxr_pack_half2(srnxr_sin(w0 * (c0[0] + bias0), sine_mode),
                                       srnxr_sin(w0 * (c0[1] + bias1), sine_mode));
        next[ot][1] = srnxr_pack_half2(srnxr_sin(w0 * (c0[2] + bias0), sine_mode),
                                       srnxr_sin(w0 * (c0[3] + bias1), sine_mode));
        next[ot][2] = srnxr_pack_half2(srnxr_sin(w0 * (c1[0] + bias8), sine_mode),
                                       srnxr_sin(w0 * (c1[1] + bias9), sine_mode));
        next[ot][3] = srnxr_pack_half2(srnxr_sin(w0 * (c1[2] + bias8), sine_mode),
                                       srnxr_sin(w0 * (c1[3] + bias9), sine_mode));
      }
      /* The buffer is overwritten by the staging two iterations ahead. */
      __syncthreads();
    }
    if (active) {
#  pragma unroll
      for (int kt = 0; kt < KT; kt++) {
#  pragma unroll
        for (int j = 0; j < 4; j++) {
          a[kt][j] = next[kt][j];
        }
      }
    }
    mma += (W / 8) * KT * 32;
    biases += W;
  }

  /* Linear head, one 8 column output tile (4 KB, read from global memory). */
  if (active) {
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    {
      uint2 bf[KT];
#  pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        bf[kt] = mma[kt * 32 + lane];
      }
#  pragma unroll
      for (int kt = 0; kt < KT; kt++) {
        srnxr_mma(c, a[kt], bf[kt]);
      }
    }
    const uint32_t output_dim = hdr->output_dim;
    const uint32_t k0 = 2 * t;
    const float hb0 = (k0 < output_dim) ? biases[k0] : 0.0f;
    const float hb1 = (k0 + 1 < output_dim) ? biases[k0 + 1] : 0.0f;
    c[0] += hb0;
    c[1] += hb1;
    c[2] += hb0;
    c[3] += hb1;

    /* Row r of the output tile is in lanes 4 (r & 7) + k / 2, element (r < 8 ? 0 : 2) + k % 2. */
    const uint32_t rank = __popc(rows & ((1u << lane) - 1u));
    const bool upper = rank >= 8;
    for (uint32_t kp = 0; kp < (output_dim + 1) / 2; kp++) {
      const uint32_t src = (rank & 7) * 4 + kp;
      const float v0 = __shfl_sync(0xFFFFFFFFu, c[0], src);
      const float v1 = __shfl_sync(0xFFFFFFFFu, c[1], src);
      const float v2 = __shfl_sync(0xFFFFFFFFu, c[2], src);
      const float v3 = __shfl_sync(0xFFFFFFFFu, c[3], src);
      if (2 * kp < output_dim) {
        y[2 * kp] = upper ? v2 : v0;
      }
      if (2 * kp + 1 < output_dim) {
        y[2 * kp + 1] = upper ? v3 : v1;
      }
    }
  }
}

#endif /* SRNXR_HAS_MMA */

#ifdef SRNXR_HAS_WARP_INTRINSICS

/* Rows of one warp-cooperative pass of the warp-cooperative SIMT path. */
#  define SRNXR_WARP_SIMT_ROWS 4

// ----------------------------------------------------------------------------
// Partially active warps
// ----------------------------------------------------------------------------
// In the Cycles volume integrator the warps are often not fully active at the image lookup
// (paths that left the volume loop, divergent branches), which excludes mma.sync and the
// lane-owned layout of srnxr_v2_eval_network_warp(). The active lanes (at least
// SRNXR_PARTIAL_MIN_LANES) then share the work of one network through a per-warp shared
// memory slice holding the activations of SRNXR_PARTIAL_ROWS queries: active lane of rank r
// computes the neuron pairs r, r + n, r + 2n, ... of every layer from the transposed FP16
// weights, the output layer is one dot product per (query, output) in index order. Every sum
// is the bias plus the inputs in index order, as in the per-thread path, so CUDA_FUSED keeps
// its FP32 arithmetic.

#  define SRNXR_PARTIAL_ROWS 2
#  define SRNXR_PARTIAL_MIN_LANES 8
#  ifndef SRNXR_SMEM_WARPS
#    ifdef GPU_KERNEL_BLOCK_NUM_THREADS
/* Warps of a Cycles kernel block. */
#      define SRNXR_SMEM_WARPS (GPU_KERNEL_BLOCK_NUM_THREADS / 32)
#    else
#      define SRNXR_SMEM_WARPS 12
#    endif
#  endif
#  define SRNXR_SMEM_WARP_FLOATS \
    (SRNXR_PARTIAL_ROWS * SRNXR_V2_MAX_WIDTH + SRNXR_PARTIAL_ROWS * SRNXR_V2_MAX_OUTPUTS)

/* One allocation per kernel, shared by every instantiation of the partial path. */
ccl_device_forceinline float *srnxr_smem_warp()
{
  __shared__ float srnxr_smem[SRNXR_SMEM_WARPS * SRNXR_SMEM_WARP_FLOATS];
  return srnxr_smem + (threadIdx.x >> 5) * SRNXR_SMEM_WARP_FLOATS;
}

ccl_device_forceinline bool srnxr_smem_available()
{
  return blockDim.y == 1 && blockDim.z == 1 && (threadIdx.x >> 5) < SRNXR_SMEM_WARPS;
}

/* KP: neuron pairs per lane, (W / 2) / KP is the smallest supported number of active lanes. */
template<int W, int KP>
ccl_device_noinline void srnxr_v2_eval_network_partial(const ccl_global uint8_t *base,
                                                       const uint32_t network,
                                                       const uint32_t mask,
                                                       const uint32_t rows,
                                                       const ccl_private float *x,
                                                       ccl_private float *y)
{
  constexpr int G = SRNXR_PARTIAL_ROWS;
  constexpr int PAIRS = W / 2;
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNetworkV2 &net = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      base + hdr->networks_offset)[network];
  const ccl_global uint16_t *weights = reinterpret_cast<const ccl_global uint16_t *>(
                                           base + hdr->weights_offset) +
                                       net.weight_offset;
  const ccl_global float *biases = reinterpret_cast<const ccl_global float *>(
                                       base + hdr->biases_offset) +
                                   net.bias_offset;
  const ccl_global uint32_t *wt_base = reinterpret_cast<const ccl_global uint32_t *>(
      reinterpret_cast<const ccl_global uint16_t *>(base + hdr->warp_weights_offset) +
      net.warp_offset);
  const uint32_t in_dim = hdr->input_dim;
  const uint32_t output_dim = hdr->output_dim;
  const uint32_t layers = hdr->hidden_layer_count;
  const uint32_t sine_mode = hdr->sine_mode;
  const uint32_t lane = srnxr_lane_id();
  const uint32_t lt = (1u << lane) - 1u;
  const uint32_t n = __popc(mask);
  const uint32_t rank = __popc(mask & lt);
  const uint32_t nrows = __popc(rows);
  const uint32_t my_row = __popc(rows & lt);
  const bool owner = (rows >> lane) & 1u;
  const ccl_global uint16_t *head = weights + W * in_dim + (layers - 1) * W * W;
  const ccl_global float *head_bias = biases + layers * W;
  float *sa = srnxr_smem_warp();
  float *so = sa + G * SRNXR_V2_MAX_WIDTH;

  for (uint32_t r0 = 0; r0 < nrows; r0 += G) {
    float xs[G][4];
#  pragma unroll
    for (int s = 0; s < G; s++) {
      const uint32_t src = srnxr_nth_lane(rows, r0 + s);
#  pragma unroll
      for (int d = 0; d < 4; d++) {
        const float v = __shfl_sync(mask, x[d], src & 31);
        xs[s][d] = (src < 32) ? v : 0.0f;
      }
    }

    /* First layer */
    {
      const float w0 = hdr->layer_w0[0];
#  pragma unroll
      for (int k = 0; k < KP; k++) {
        const uint32_t pair = rank + n * k;
        if (pair < PAIRS) {
#  pragma unroll
          for (int e = 0; e < 2; e++) {
            const uint32_t o = 2 * pair + e;
            const ccl_global uint16_t *row = weights + o * in_dim;
#  pragma unroll
            for (int s = 0; s < G; s++) {
              float sum = biases[o];
              for (uint32_t d = 0; d < in_dim; d++) {
                sum += srnxr_half_to_float(row[d]) * xs[s][d];
              }
              sa[s * W + o] = srnxr_sin(w0 * sum, sine_mode);
            }
          }
        }
      }
    }
    __syncwarp(mask);

    /* Hidden layers */
    const ccl_global uint32_t *wt = wt_base;
    const ccl_global float *bl = biases + W;
    /* Hidden layers: lane of rank r owns the contiguous neuron pairs [r KP, r KP + KP), read as
     * 16 (or 8) byte words of the transposed weights. PAIRS is a multiple of KP, so a lane owns
     * a whole block or nothing. */
    const uint32_t base_pair = rank * KP;
    const bool owns_pairs = base_pair < PAIRS;
    for (uint32_t layer = 1; layer < layers; layer++) {
      float acc[G][KP][2];
#  pragma unroll
      for (int k = 0; k < KP; k++) {
        const uint32_t pair = min(base_pair + k, uint32_t(PAIRS - 1));
#  pragma unroll
        for (int s = 0; s < G; s++) {
          acc[s][k][0] = bl[2 * pair];
          acc[s][k][1] = bl[2 * pair + 1];
        }
      }
      if (owns_pairs) {
        for (int i = 0; i < W; i++) {
          float v[G];
#  pragma unroll
          for (int s = 0; s < G; s++) {
            v[s] = sa[s * W + i];
          }
          const ccl_global uint32_t *wrow = wt + i * PAIRS + base_pair;
          uint32_t wq[KP];
          if (KP % 4 == 0) {
#  pragma unroll
            for (int k = 0; k < KP; k += 4) {
              const uint4 w4 = *reinterpret_cast<const ccl_global uint4 *>(wrow + k);
              wq[k] = w4.x;
              wq[k + 1] = w4.y;
              wq[k + 2] = w4.z;
              wq[k + 3] = w4.w;
            }
          }
          else {
#  pragma unroll
            for (int k = 0; k < KP; k += 2) {
              const uint2 w2 = *reinterpret_cast<const ccl_global uint2 *>(wrow + k);
              wq[k] = w2.x;
              wq[k + 1] = w2.y;
            }
          }
#  pragma unroll
          for (int k = 0; k < KP; k++) {
            const float lo = srnxr_half_lo(wq[k]), hi = srnxr_half_hi(wq[k]);
#  pragma unroll
            for (int s = 0; s < G; s++) {
              acc[s][k][0] += lo * v[s];
              acc[s][k][1] += hi * v[s];
            }
          }
        }
      }
      /* Every lane has read the previous activations before any is overwritten. */
      __syncwarp(mask);
      const float w0 = hdr->layer_w0[layer];
      if (owns_pairs) {
#  pragma unroll
        for (int k = 0; k < KP; k++) {
          const uint32_t pair = base_pair + k;
#  pragma unroll
          for (int s = 0; s < G; s++) {
            sa[s * W + 2 * pair] = srnxr_sin(w0 * acc[s][k][0], sine_mode);
            sa[s * W + 2 * pair + 1] = srnxr_sin(w0 * acc[s][k][1], sine_mode);
          }
        }
      }
      __syncwarp(mask);
      wt += W * PAIRS;
      bl += W;
    }

    /* Output layer: one dot product per (query, output), index order. */
    for (uint32_t q = rank; q < G * output_dim; q += n) {
      const uint32_t s = q / output_dim, k = q % output_dim;
      if (r0 + s < nrows) {
        so[s * SRNXR_V2_MAX_OUTPUTS + k] = srnxr_dot_row(head + k * W, sa + s * W, W, head_bias[k]);
      }
    }
    __syncwarp(mask);
    if (owner && my_row >= r0 && my_row < r0 + G) {
      for (uint32_t k = 0; k < output_dim; k++) {
        y[k] = so[(my_row - r0) * SRNXR_V2_MAX_OUTPUTS + k];
      }
    }
    __syncwarp(mask);
  }
}

/* Decode the queries of the active lanes of a warp together, with the backend of the header:
 * CUDA_TENSOR (fully active warps only), the warp-cooperative SIMT path (CUDA_FUSED, fully
 * active warps) or the partial-warp path (any backend, at least SRNXR_PARTIAL_MIN_LANES active
 * lanes; a counted fallback for CUDA_TENSOR). Lanes without a query pass active = false.
 * Returns false without doing anything when too few lanes are active. */
ccl_device_inline bool srnxr_v2_decode_warp(const ccl_global uint8_t *base,
                                            const float p[4],
                                            const bool active,
                                            const uint32_t leaf,
                                            ccl_private SRNXRBlend &blend)
{
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const uint32_t mask = __activemask();
  const bool full = mask == 0xFFFFFFFFu;
  /* Convergence statistics: calls per warp and active lanes. */
  srnxr_counter_add(base, SRNXR_COUNTER_WARP_CALLS, srnxr_lane_id() == uint32_t(__ffs(mask) - 1));
  srnxr_counter_add(base, SRNXR_COUNTER_WARP_ACTIVE_LANES, 1u);
  const bool tensor_backend = hdr->backend == SRNXR_BACKEND_CUDA_TENSOR;
  const bool partial_ok = (hdr->flags & SRNXR_DEVICE_FLAG_WARP_WEIGHTS) &&
                          (hdr->flags & SRNXR_DEVICE_FLAG_PARTIAL_WARPS) &&
                          __popc(mask) >= SRNXR_PARTIAL_MIN_LANES && srnxr_smem_available();
  if (!full && !partial_ok) {
    return false;
  }
  const ccl_global SRNXRLeafV2 *leaves = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
      base + hdr->leaves_offset);
  const ccl_global uint32_t *neighbors = reinterpret_cast<const ccl_global uint32_t *>(
      base + hdr->neighbors_offset);
  const ccl_global SRNXRNetworkV2 *networks =
      reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(base + hdr->networks_offset);
  /* mma.sync needs every lane of the warp. */
  const bool tensor = tensor_backend && full && (hdr->flags & SRNXR_DEVICE_FLAG_MMA_WEIGHTS);
  const uint32_t lane = srnxr_lane_id();
  const uint32_t leader_lane = __ffs(mask) - 1;

  /* Pending candidates as a bit mask over the first 64 neighbor positions. */
  uint32_t begin = 0, end = 0;
  uint64_t pending = 0;
  if (active) {
    begin = leaves[leaf].neighbor_begin;
    end = leaves[leaf].neighbor_end;
    const uint32_t n = min(end - begin, 64u);
    for (uint32_t i = 0; i < n; i++) {
      if (srnxr_v2_in_expanded(leaves[neighbors[begin + i]], p)) {
        pending |= uint64_t(1) << i;
      }
    }
  }

  uint32_t tiles = 0, tile_rows = 0, passes = 0, evals = 0, warp_evals = 0, partial_evals = 0,
           fallback = 0, simt = 0;
  for (;;) {
    const uint32_t waiting = __ballot_sync(mask, pending != 0);
    if (waiting == 0) {
      break;
    }
    /* The lowest waiting lane picks the network of its first pending candidate. */
    const uint32_t leader = __ffs(waiting) - 1;
    uint32_t my_network = SRNXR_INVALID_INDEX;
    if (pending) {
      my_network = leaves[neighbors[begin + __ffsll((long long)pending) - 1]].network;
    }
    const uint32_t network = __shfl_sync(mask, my_network, leader);

    /* Join with a pending candidate that uses the same network. */
    int my_pos = -1;
    for (uint64_t bits = pending; bits; bits &= bits - 1) {
      const int i = __ffsll((long long)bits) - 1;
      if (leaves[neighbors[begin + i]].network == network) {
        my_pos = i;
        break;
      }
    }
    const uint32_t members = __ballot_sync(mask, my_pos >= 0);
    const uint32_t rank = __popc(members & ((1u << lane) - 1u));
    const uint32_t max_rows = tensor ? 16u : 32u;
    const uint32_t rows = (uint32_t(__popc(members)) <= max_rows) ?
                              members :
                              members & ((1u << srnxr_nth_lane(members, max_rows)) - 1u);
    const bool in_pass = my_pos >= 0 && rank < max_rows;

    const ccl_global SRNXRLeafV2 *cand = in_pass ? &leaves[neighbors[begin + my_pos]] : nullptr;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (in_pass) {
      srnxr_v2_local_input(*cand, p, x);
    }
    float y[SRNXR_V2_MAX_OUTPUTS];
    const uint32_t width = networks[network].width;
    bool done = false;
#  ifdef SRNXR_HAS_MMA
    if (tensor && networks[network].mma_offset != SRNXR_INVALID_INDEX) {
      if (width == 256) {
        srnxr_v2_eval_network_tensor<256>(base, network, rows, x, y);
        done = true;
      }
      else if (width == 128) {
        srnxr_v2_eval_network_tensor<128>(base, network, rows, x, y);
        done = true;
      }
      else if (width == 64) {
        srnxr_v2_eval_network_tensor<64>(base, network, rows, x, y);
        done = true;
      }
      if (done && lane == leader_lane) {
        tiles++;
        tile_rows += __popc(rows);
      }
    }
#  endif
    const bool has_warp_copy = networks[network].warp_offset != SRNXR_INVALID_INDEX;
#  ifdef SRNXR_HAS_WARP_SIMT
    if (!done && !tensor_backend && full && has_warp_copy) {
      if (width == 256) {
        srnxr_v2_eval_network_warp<256, SRNXR_WARP_SIMT_ROWS>(base, network, rows, x, y);
        done = true;
      }
      else if (width == 128) {
        srnxr_v2_eval_network_warp<128, SRNXR_WARP_SIMT_ROWS>(base, network, rows, x, y);
        done = true;
      }
      else if (width == 64) {
        srnxr_v2_eval_network_warp<64, SRNXR_WARP_SIMT_ROWS>(base, network, rows, x, y);
        done = true;
      }
      if (done && lane == leader_lane) {
        passes += (__popc(rows) + SRNXR_WARP_SIMT_ROWS - 1) / SRNXR_WARP_SIMT_ROWS;
      }
      if (done && in_pass) {
        warp_evals++;
      }
    }
#  endif
    if (!done && partial_ok && has_warp_copy) {
      const bool wide = __popc(mask) >= 16;
      if (width == 256) {
        if (wide) {
          srnxr_v2_eval_network_partial<256, 8>(base, network, mask, rows, x, y);
        }
        else {
          srnxr_v2_eval_network_partial<256, 16>(base, network, mask, rows, x, y);
        }
        done = true;
      }
      else if (width == 128) {
        if (wide) {
          srnxr_v2_eval_network_partial<128, 4>(base, network, mask, rows, x, y);
        }
        else {
          srnxr_v2_eval_network_partial<128, 8>(base, network, mask, rows, x, y);
        }
        done = true;
      }
      else if (width == 64) {
        if (wide) {
          srnxr_v2_eval_network_partial<64, 2>(base, network, mask, rows, x, y);
        }
        else {
          srnxr_v2_eval_network_partial<64, 4>(base, network, mask, rows, x, y);
        }
        done = true;
      }
      if (done && in_pass) {
        partial_evals++;
        if (tensor_backend) {
          fallback++;
        }
      }
    }
    if (!done && in_pass) {
      /* No specialization for this network: per-thread SIMT. For CUDA_FUSED this is the
       * same arithmetic, for CUDA_TENSOR it is a counted fallback. */
      srnxr_v2_eval_network_simt(base, network, x, y);
      simt++;
      if (tensor_backend) {
        fallback++;
      }
    }
    if (in_pass) {
      srnxr_v2_leaf_output(*cand, y, hdr->output_dim);
      srnxr_blend_add(blend, srnxr_v2_blend_exponent(*cand, p), y, hdr->output_dim);
      pending &= ~(uint64_t(1) << my_pos);
      evals++;
    }
  }

  srnxr_counter_add(base, SRNXR_COUNTER_TENSOR_TILES, tiles);
  srnxr_counter_add(base, SRNXR_COUNTER_TENSOR_ROWS, tile_rows);
  srnxr_counter_add(base, SRNXR_COUNTER_WARP_SIMT_PASSES, passes);
  srnxr_counter_add(base, SRNXR_COUNTER_WARP_SIMT_EVALS, warp_evals);
  srnxr_counter_add(base, SRNXR_COUNTER_PARTIAL_WARP_EVALS, partial_evals);
  srnxr_counter_add(base, SRNXR_COUNTER_NETWORK_EVALS, evals);
  srnxr_counter_add(base, SRNXR_COUNTER_SIMT_EVALS, simt);
  srnxr_counter_add(base, SRNXR_COUNTER_FALLBACK_SIMT, fallback);

  /* Neighbor positions beyond the 64 bit mask. */
  if (active && end - begin > 64) {
    srnxr_v2_decode_simt_range(
        base,
        p,
        begin + 64,
        end,
        blend,
        tensor_backend ? SRNXR_COUNTER_FALLBACK_SIMT : SRNXR_COUNTER_SIMT_EVALS);
  }
  return true;
}

#endif /* SRNXR_HAS_WARP_INTRINSICS */


/* Conservative range of the mapped channels in the box p +- bake_margin over the shutter time
 * interval of p[3]. The value at a point is a convex combination of the outputs of the leaves
 * whose expanded box contains the point, all of them neighbors of the strict leaf of the point:
 * the union of the certified output ranges of the neighbors (of the strict leaves the box touches)
 * whose expanded box intersects the box bounds every value in the box. */
ccl_device_inline void srnxr_v2_bound(const ccl_global uint8_t *base,
                                      const float p[4],
                                      ccl_private float *lo,
                                      ccl_private float *hi)
{
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const ccl_global SRNXRNodeV2 *nodes = reinterpret_cast<const ccl_global SRNXRNodeV2 *>(
      base + hdr->nodes_offset);
  const ccl_global SRNXRLeafV2 *leaves = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
      base + hdr->leaves_offset);
  const uint32_t channels = hdr->channel_count;
  for (uint32_t c = 0; c < channels; c++) {
    lo[c] = FLT_MAX;
    hi[c] = -FLT_MAX;
  }
  const ccl_global uint32_t *neighbors = reinterpret_cast<const ccl_global uint32_t *>(
      base + hdr->neighbors_offset);
  float qmin[4], qmax[4];
  for (int d = 0; d < 3; d++) {
    qmin[d] = p[d] - hdr->bake_margin[d];
    qmax[d] = p[d] + hdr->bake_margin[d];
  }
  qmin[3] = p[3] - 0.5f * fabsf(hdr->time_shutter);
  qmax[3] = p[3] + 0.5f * fabsf(hdr->time_shutter);
  uint32_t stack[64];
  int top = 0;
  stack[top++] = 0;
  while (top > 0) {
    const ccl_global SRNXRNodeV2 &node = nodes[stack[--top]];
    if (node.bmin[0] > qmax[0] || node.bmin[1] > qmax[1] || node.bmin[2] > qmax[2] ||
        node.bmax[0] < qmin[0] || node.bmax[1] < qmin[1] || node.bmax[2] < qmin[2])
    {
      continue;
    }
    if (node.first_child == SRNXR_INVALID_INDEX) {
      const ccl_global SRNXRLeafV2 &strict = leaves[node.leaf];
      for (uint32_t i = strict.neighbor_begin; i < strict.neighbor_end; i++) {
        const ccl_global SRNXRLeafV2 &leaf = leaves[neighbors[i]];
        bool touches = true;
        for (int d = 0; d < 4; d++) {
          touches = touches && leaf.expanded_min[d] <= qmax[d] && leaf.expanded_max[d] >= qmin[d];
        }
        if (!touches) {
          continue;
        }
        for (uint32_t c = 0; c < channels; c++) {
          lo[c] = fminf(lo[c], leaf.bound_min[c]);
          hi[c] = fmaxf(hi[c], leaf.bound_max[c]);
        }
      }
      continue;
    }
    if (top + 8 > 64) {
      /* Stack exhausted, fall back to the root range: the range of every leaf. */
      for (uint32_t i = 0; i < hdr->leaf_count; i++) {
        for (uint32_t c = 0; c < channels; c++) {
          lo[c] = fminf(lo[c], leaves[i].bound_min[c]);
          hi[c] = fmaxf(hi[c], leaves[i].bound_max[c]);
        }
      }
      return;
    }
    for (uint32_t k = 0; k < 8; k++) {
      stack[top++] = node.first_child + k;
    }
  }
}

/* All fields of the blended field at p (object space position and simulation time). Returns
 * false outside the domain, where out is outside_value. */
ccl_device_inline bool srnxr_v2_decode(const ccl_global uint8_t *base,
                                       const float p[4],
                                       const bool active,
                                       ccl_private float *out)
{
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  bool inside = active;
  for (int d = 0; d < 4; d++) {
    inside = inside && p[d] >= hdr->domain_min[d] && p[d] <= hdr->domain_max[d];
  }
  const uint32_t leaf = inside ? srnxr_v2_find_leaf(base, p) : SRNXR_INVALID_INDEX;
  inside = inside && leaf != SRNXR_INVALID_INDEX;
  if (active) {
    srnxr_counter_add(base, inside ? SRNXR_COUNTER_QUERIES : SRNXR_COUNTER_OUTSIDE, 1);
  }

  SRNXRBlend blend;
  srnxr_blend_init(blend);

  bool done = false;
#ifdef SRNXR_HAS_WARP_INTRINSICS
  if ((hdr->backend == SRNXR_BACKEND_CUDA_TENSOR &&
       (hdr->flags & (SRNXR_DEVICE_FLAG_MMA_WEIGHTS | SRNXR_DEVICE_FLAG_WARP_WEIGHTS))) ||
      (hdr->backend == SRNXR_BACKEND_CUDA_FUSED && (hdr->flags & SRNXR_DEVICE_FLAG_WARP_WEIGHTS)))
  {
    /* Lanes outside the domain take part with an empty candidate list. */
    done = srnxr_v2_decode_warp(base, p, inside, leaf, blend);
  }
#endif
  if (!done && inside) {
    const ccl_global SRNXRLeafV2 &l = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
        base + hdr->leaves_offset)[leaf];
    srnxr_v2_decode_simt_range(base,
                               p,
                               l.neighbor_begin,
                               l.neighbor_end,
                               blend,
                               (hdr->backend == SRNXR_BACKEND_CUDA_FUSED) ?
                                   SRNXR_COUNTER_SIMT_EVALS :
                                   SRNXR_COUNTER_FALLBACK_SIMT);
  }

  if (!inside) {
    for (uint32_t k = 0; k < hdr->output_dim; k++) {
      out[k] = hdr->outside_value;
    }
    return false;
  }
  /* The strict leaf always contributes, so weight_sum >= 1. */
  const float inv = srnxr_div(1.0f, blend.weight_sum);
  for (uint32_t k = 0; k < hdr->output_dim; k++) {
    out[k] = blend.value[k] * inv;
  }
  return true;
}

CCL_NAMESPACE_END
