/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Wavefront (queued) decoding in the standalone bench: every point is one launch slot with one
 * lookup, the stages of kernel/util/image_3d_srnxr_queue.h run as separate kernels and are
 * timed one by one. Included by srnxr_bench.cu. */

#pragma once

struct BlobBase {
  const uint8_t *blob;
  __device__ __forceinline__ const uint8_t *operator()(uint32_t /*slot*/) const
  {
    return blob;
  }
};

__global__ void wf_record_kernel(const SRNXRQueue q,
                                 const float4 *points,
                                 const int n,
                                 const uint32_t key)
{
#ifdef __CUDA_ARCH__
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float4 pt = points[i];
  const float p[4] = {pt.x, pt.y, pt.z, pt.w};
  uint32_t cursor = 0;
  srnxr_queue_record(q, uint32_t(i), cursor, p, 0u, key);
  q.window_count[i] = cursor;
#endif
}

__global__ void wf_expand_kernel(const SRNXRQueue q, const BlobBase base_of)
{
#ifdef __CUDA_ARCH__
  const uint32_t n = srnxr_queue_num_requests(q);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    srnxr_queue_expand(q, i, base_of);
  }
#endif
}

__global__ void wf_scan_kernel(const SRNXRQueue q)
{
#ifdef __CUDA_ARCH__
  srnxr_queue_scan_block(q);
#endif
}

__global__ void wf_scatter_kernel(const SRNXRQueue q)
{
#ifdef __CUDA_ARCH__
  const uint32_t n = srnxr_queue_num_requests(q);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    srnxr_queue_scatter(q, i);
  }
#endif
}

__global__ void wf_tensor_kernel(const SRNXRQueue q, const BlobBase base_of)
{
#ifdef __CUDA_ARCH__
  const uint32_t tiles = q.counters[SRNXR_QUEUE_COUNTER_TILES];
  const uint32_t warps_per_block = blockDim.x >> 5;
  const uint32_t warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const uint32_t stride = gridDim.x * warps_per_block;
  for (uint32_t t = warp; t < tiles; t += stride) {
    srnxr_queue_tensor_tile(q, t, base_of);
  }
#endif
}

template<int NT>
__global__ void wf_tensor_block_kernel(const SRNXRQueue q, const BlobBase base_of)
{
#ifdef __CUDA_ARCH__
  __shared__ __align__(16) uint8_t smem[SRNXR_TENSOR_BLOCK_SMEM_BYTES + 16];
  const uint32_t tiles = q.counters[SRNXR_QUEUE_COUNTER_TILES];
  const uint32_t tiles_per_block = (blockDim.x >> 5) * NT;
  for (uint32_t first = blockIdx.x * tiles_per_block; first < tiles;
       first += gridDim.x * tiles_per_block)
  {
    srnxr_queue_tensor_group<NT>(q, first, tiles, base_of, smem);
  }
#endif
}

__global__ void wf_blend_kernel(const SRNXRQueue q, const BlobBase base_of)
{
#ifdef __CUDA_ARCH__
  const uint32_t n = srnxr_queue_num_requests(q);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    srnxr_queue_blend(q, i, base_of);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    srnxr_queue_finish_stats(q);
  }
#endif
}

__global__ void wf_replay_kernel(const SRNXRQueue q,
                                 const uint8_t *blob,
                                 const float4 *points,
                                 const int n,
                                 const uint32_t key,
                                 const int output_dim,
                                 float *out)
{
#ifdef __CUDA_ARCH__
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = i < n;
  float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (active) {
    const float4 pt = points[i];
    p[0] = pt.x;
    p[1] = pt.y;
    p[2] = pt.z;
    p[3] = pt.w;
  }
  float v[SRNXR_V2_MAX_OUTPUTS];
  uint32_t hint = 0;
  const bool hit = active &&
                   srnxr_queue_replay(q, uint32_t(i), q.window_count[i], hint, p, key, output_dim, v);
  /* Misses decode inline, every lane takes part (warp-cooperative decoders). */
  if (__any_sync(__activemask(), active && !hit)) {
    float w[SRNXR_V2_MAX_OUTPUTS];
    srnxr_v2_decode(blob, p, active && !hit, w);
    if (active && !hit) {
      for (int k = 0; k < output_dim; k++) {
        v[k] = w[k];
      }
    }
  }
  if (active) {
    for (int k = 0; k < output_dim; k++) {
      out[size_t(i) * output_dim + k] = v[k];
    }
  }
#endif
}

struct WavefrontQueue {
  SRNXRQueue q;
  std::vector<void *> allocations;

  void *alloc(const size_t bytes)
  {
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemset(p, 0, bytes));
    allocations.push_back(p);
    return p;
  }

  void init(const uint32_t capacity, const uint32_t num_buckets, const uint32_t num_slots)
  {
    memset(&q, 0, sizeof(q));
    q.capacity = capacity;
    q.num_buckets = num_buckets;
    q.k_max = 1;
    q.num_slots = num_slots;
    q.req_p = (float *)alloc(size_t(capacity) * 4 * sizeof(float));
    q.req_slot = (uint32_t *)alloc(size_t(capacity) * sizeof(uint32_t));
    q.req_key = (uint32_t *)alloc(size_t(capacity) * sizeof(uint32_t));
    q.req_info = (uint32_t *)alloc(size_t(capacity) * sizeof(uint32_t));
    q.req_out = (float *)alloc(size_t(capacity) * SRNXR_QUEUE_OUT_STRIDE * sizeof(float));
    const size_t pairs = size_t(capacity) * SRNXR_QUEUE_MAX_PAIRS;
    q.pair_leaf = (uint32_t *)alloc(pairs * sizeof(uint32_t));
    q.pair_bucket = (uint32_t *)alloc(pairs * sizeof(uint32_t));
    q.pair_exp = (float *)alloc(pairs * sizeof(float));
    q.pair_out = (float *)alloc(pairs * SRNXR_QUEUE_OUT_STRIDE * sizeof(float));
    q.sorted = (uint32_t *)alloc(pairs * sizeof(uint32_t));
    q.bucket_count = (uint32_t *)alloc(size_t(num_buckets) * sizeof(uint32_t));
    q.bucket_offset = (uint32_t *)alloc(size_t(num_buckets + 1) * sizeof(uint32_t));
    q.bucket_tile = (uint32_t *)alloc(size_t(num_buckets + 1) * sizeof(uint32_t));
    q.bucket_cursor = (uint32_t *)alloc(size_t(num_buckets) * sizeof(uint32_t));
    q.window = (uint32_t *)alloc(size_t(num_slots) * q.k_max * sizeof(uint32_t));
    q.window_count = (uint32_t *)alloc(size_t(num_slots) * sizeof(uint32_t));
    q.counters = (uint32_t *)alloc(SRNXR_QUEUE_COUNTER_NUM * sizeof(uint32_t));
    q.stats = (unsigned long long int *)alloc(SRNXR_QUEUE_COUNTER_NUM * sizeof(uint64_t));
  }

  void reset()
  {
    CUDA_CHECK(cudaMemsetAsync(q.counters, 0, SRNXR_QUEUE_COUNTER_NUM * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(q.bucket_count, 0, size_t(q.num_buckets) * sizeof(uint32_t)));
  }

  void free()
  {
    for (void *p : allocations) {
      cudaFree(p);
    }
    allocations.clear();
  }
};

/* One wavefront decode of the first count points. Records the stage times (ms) into times[7]
 * (record, expand, scan, scatter, tensor, blend, replay) when events are given. */
static void wavefront_run(WavefrontQueue &wq,
                          const uint8_t *d_blob,
                          const float4 *d_points,
                          const int count,
                          const uint32_t key,
                          const int output_dim,
                          float *d_out,
                          const int block,
                          const int grid_persistent,
                          cudaEvent_t *events,
                          const bool block_tensor,
                          const int tiles_per_warp)
{
  const SRNXRQueue &q = wq.q;
  const BlobBase base_of = {d_blob};
  const int grid = (count + block - 1) / block;
  wq.reset();
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[0]));
  }
  wf_record_kernel<<<grid, block>>>(q, d_points, count, key);
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[1]));
  }
  wf_expand_kernel<<<grid_persistent, block>>>(q, base_of);
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[2]));
  }
  wf_scan_kernel<<<1, 1024>>>(q);
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[3]));
  }
  wf_scatter_kernel<<<grid_persistent, block>>>(q);
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[4]));
  }
  if (block_tensor && tiles_per_warp == 2) {
    wf_tensor_block_kernel<2><<<grid_persistent, block>>>(q, base_of);
  }
  else if (block_tensor) {
    wf_tensor_block_kernel<1><<<grid_persistent, block>>>(q, base_of);
  }
  else {
    wf_tensor_kernel<<<grid_persistent, block>>>(q, base_of);
  }
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[5]));
  }
  wf_blend_kernel<<<grid_persistent, block>>>(q, base_of);
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[6]));
  }
  wf_replay_kernel<<<grid, block>>>(q, d_blob, d_points, count, key, output_dim, d_out);
  if (events) {
    CUDA_CHECK(cudaEventRecord(events[7]));
  }
}
