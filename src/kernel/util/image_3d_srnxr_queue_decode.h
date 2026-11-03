/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* CUDA device code of the wavefront decode queue (kernel/util/image_3d_srnxr_queue.h): record and
 * replay of the lookups and the batch stages. Include after kernel/util/image_3d_srnxr_decode.h. */

#include "kernel/util/image_3d_srnxr_queue.h"

CCL_NAMESPACE_BEGIN

#if defined(__KERNEL_CUDA__) && defined(__CUDA_ARCH__)

/* Aggregated per warp when warp intrinsics are allowed (not inside OptiX programs). */
ccl_device_forceinline void srnxr_queue_stat_add(const SRNXRQueue &q,
                                                 const SRNXRQueueCounter counter,
                                                 const uint32_t n)
{
#  if defined(SRNXR_HAS_WARP_INTRINSICS) && (__CUDA_ARCH__ >= 800)
  const uint32_t mask = __match_any_sync(__activemask(), uint32_t(counter));
  const uint32_t total = __reduce_add_sync(mask, n);
  uint32_t lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  if (lane == uint32_t(__ffs(mask) - 1) && total) {
    atomicAdd(q.stats + counter, (unsigned long long int)total);
  }
#  else
  if (n) {
    atomicAdd(q.stats + counter, (unsigned long long int)n);
  }
#  endif
}

/* Allocate n consecutive request indices, aggregated per warp where possible. */
ccl_device_forceinline uint32_t srnxr_queue_alloc(const SRNXRQueue &q, const uint32_t n)
{
#  if defined(SRNXR_HAS_WARP_INTRINSICS)
  const uint32_t mask = __activemask();
  uint32_t lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  const uint32_t leader = __ffs(mask) - 1;
  /* Inclusive prefix over the active lanes. */
  uint32_t prefix = 0, total = 0;
  for (uint32_t bits = mask; bits; bits &= bits - 1) {
    const uint32_t l = __ffs(bits) - 1;
    const uint32_t v = __shfl_sync(mask, n, l);
    if (l < lane) {
      prefix += v;
    }
    total += v;
  }
  uint32_t base = 0;
  if (lane == leader) {
    /* Saturate at the capacity: a full queue stops counting (the counter never wraps). */
    base = (q.counters[SRNXR_QUEUE_COUNTER_REQUESTS] >= q.capacity) ?
               q.capacity :
               atomicAdd(q.counters + SRNXR_QUEUE_COUNTER_REQUESTS, total);
  }
  base = __shfl_sync(mask, base, leader);
  return base + prefix;
#  else
  return (q.counters[SRNXR_QUEUE_COUNTER_REQUESTS] >= q.capacity) ?
             q.capacity :
             atomicAdd(q.counters + SRNXR_QUEUE_COUNTER_REQUESTS, n);
#  endif
}

/* Record a lookup of the image with info id slot (decode key key) at p. cursor is the number
 * of lookups recorded for the launch slot so far. Returns false when it could not be recorded
 * (the window or the queue is full), the caller then decodes inline. */
ccl_device_inline bool srnxr_queue_record(const SRNXRQueue &q,
                                          const uint32_t launch_slot,
                                          ccl_private uint32_t &cursor,
                                          const float p[4],
                                          const uint32_t image_slot,
                                          const uint32_t key)
{
  if (cursor >= q.k_max) {
    srnxr_queue_stat_add(q, SRNXR_QUEUE_COUNTER_OVERFLOW, 1);
    return false;
  }
  const uint32_t idx = srnxr_queue_alloc(q, 1);
  if (idx >= q.capacity) {
    srnxr_queue_stat_add(q, SRNXR_QUEUE_COUNTER_OVERFLOW, 1);
    return false;
  }
  q.req_p[idx * 4 + 0] = p[0];
  q.req_p[idx * 4 + 1] = p[1];
  q.req_p[idx * 4 + 2] = p[2];
  q.req_p[idx * 4 + 3] = p[3];
  q.req_slot[idx] = image_slot;
  q.req_key[idx] = key;
  q.window[size_t(cursor) * q.num_slots + launch_slot] = idx;
  cursor++;
  return true;
}

/* Look a query up in the window of the launch slot: exact position and model, searched from the
 * hint (the lookup after the previous hit). Returns false when the query is not there or its
 * request could not be served (incomplete), the caller decodes inline. */
ccl_device_inline bool srnxr_queue_replay(const SRNXRQueue &q,
                                          const uint32_t launch_slot,
                                          const uint32_t count,
                                          ccl_private uint32_t &hint,
                                          const float p[4],
                                          const uint32_t key,
                                          const uint32_t output_dim,
                                          ccl_private float *out)
{
  for (uint32_t n = 0; n < count; n++) {
    uint32_t k = hint + n;
    k = (k >= count) ? k - count : k;
    const uint32_t idx = q.window[size_t(k) * q.num_slots + launch_slot];
    if (q.req_key[idx] != key) {
      continue;
    }
    const ccl_global float *rp = q.req_p + idx * 4;
    if (rp[0] != p[0] || rp[1] != p[1] || rp[2] != p[2] || rp[3] != p[3]) {
      continue;
    }
    hint = k + 1;
    if (q.req_info[idx] & SRNXR_QUEUE_REQUEST_INCOMPLETE) {
      srnxr_queue_stat_add(q, SRNXR_QUEUE_COUNTER_REPLAY_MISS, 1);
      return false;
    }
    const ccl_global float *ro = q.req_out + idx * SRNXR_QUEUE_OUT_STRIDE;
    for (uint32_t c = 0; c < output_dim; c++) {
      out[c] = ro[c];
    }
    srnxr_queue_stat_add(q, SRNXR_QUEUE_COUNTER_REPLAY_HITS, 1);
    return true;
  }
  srnxr_queue_stat_add(q, SRNXR_QUEUE_COUNTER_REPLAY_MISS, 1);
  return false;
}

/* Number of requests of the batch. */
ccl_device_forceinline uint32_t srnxr_queue_num_requests(const SRNXRQueue &q)
{
  return min(q.counters[SRNXR_QUEUE_COUNTER_REQUESTS], q.capacity);
}

/* Whether the image can be decoded through the queue (its buckets fit). */
ccl_device_forceinline bool srnxr_queue_image_eligible(const SRNXRQueue &q,
                                                       const ccl_global SRNXRDeviceHeaderV2 *hdr)
{
  return hdr->bucket_base != SRNXR_QUEUE_INVALID_BUCKET_BASE &&
         hdr->bucket_base + hdr->network_count <= q.num_buckets &&
         (hdr->flags & SRNXR_DEVICE_FLAG_MMA_WEIGHTS);
}

// ----------------------------------------------------------------------------
// Batch stages. base_of(slot) returns the blob of the image with info id slot.
// ----------------------------------------------------------------------------

/* Stage 2: expand request i into its pairs and count the buckets. */
template<typename BaseOf>
ccl_device_inline void srnxr_queue_expand(const SRNXRQueue &q, const uint32_t i, BaseOf base_of)
{
  const ccl_global uint8_t *base = base_of(q.req_slot[i]);
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const float p[4] = {q.req_p[i * 4], q.req_p[i * 4 + 1], q.req_p[i * 4 + 2], q.req_p[i * 4 + 3]};
  bool inside = true;
  for (int d = 0; d < 4; d++) {
    inside = inside && p[d] >= hdr->domain_min[d] && p[d] <= hdr->domain_max[d];
  }
  const uint32_t leaf = inside ? srnxr_v2_find_leaf(base, p) : SRNXR_INVALID_INDEX;
  inside = inside && leaf != SRNXR_INVALID_INDEX;
  uint32_t n = 0, flags = 0;
  if (!inside) {
    flags |= SRNXR_QUEUE_REQUEST_OUTSIDE;
  }
  else if (!srnxr_queue_image_eligible(q, hdr)) {
    flags |= SRNXR_QUEUE_REQUEST_INCOMPLETE;
  }
  else {
    const ccl_global SRNXRLeafV2 *leaves = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
        base + hdr->leaves_offset);
    const ccl_global uint32_t *neighbors = reinterpret_cast<const ccl_global uint32_t *>(
        base + hdr->neighbors_offset);
    const ccl_global SRNXRNetworkV2 *networks =
        reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(base + hdr->networks_offset);
    const uint32_t begin = leaves[leaf].neighbor_begin, end = leaves[leaf].neighbor_end;
    for (uint32_t j = begin; j < end; j++) {
      const uint32_t cand = neighbors[j];
      if (!srnxr_v2_in_expanded(leaves[cand], p)) {
        continue;
      }
      const uint32_t network = leaves[cand].network;
      if (n >= SRNXR_QUEUE_MAX_PAIRS || networks[network].mma_offset == SRNXR_INVALID_INDEX) {
        flags |= SRNXR_QUEUE_REQUEST_INCOMPLETE;
        break;
      }
      const uint32_t pair = i * SRNXR_QUEUE_MAX_PAIRS + n;
      const uint32_t bucket = hdr->bucket_base + network;
      q.pair_leaf[pair] = cand;
      q.pair_bucket[pair] = bucket;
      q.pair_exp[pair] = srnxr_v2_blend_exponent(leaves[cand], p);
      atomicAdd(q.bucket_count + bucket, 1u);
      n++;
    }
  }
  if (flags & SRNXR_QUEUE_REQUEST_INCOMPLETE) {
    srnxr_queue_stat_add(q, SRNXR_QUEUE_COUNTER_INCOMPLETE, 1);
  }
  q.req_info[i] = n | flags;
}

/* Stage 3: exclusive scan of the bucket counts (pairs and tiles), one block. blockDim.x must be
 * a multiple of 32 and at most 1024. */
ccl_device_inline void srnxr_queue_scan_block(const SRNXRQueue &q)
{
  __shared__ uint32_t warp_pairs[32], warp_tiles[32];
  __shared__ uint32_t carry[2];
  const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t num_warps = blockDim.x >> 5;
  if (tid == 0) {
    carry[0] = 0;
    carry[1] = 0;
  }
  __syncthreads();
  for (uint32_t b0 = 0; b0 < q.num_buckets; b0 += blockDim.x) {
    const uint32_t b = b0 + tid;
    const uint32_t c = (b < q.num_buckets) ? q.bucket_count[b] : 0;
    const uint32_t t = (c + SRNXR_QUEUE_TILE_ROWS - 1) / SRNXR_QUEUE_TILE_ROWS;
    /* Inclusive warp scans. */
    uint32_t sc = c, st = t;
    for (uint32_t o = 1; o < 32; o <<= 1) {
      const uint32_t vc = __shfl_up_sync(0xFFFFFFFFu, sc, o);
      const uint32_t vt = __shfl_up_sync(0xFFFFFFFFu, st, o);
      if (lane >= o) {
        sc += vc;
        st += vt;
      }
    }
    if (lane == 31) {
      warp_pairs[warp] = sc;
      warp_tiles[warp] = st;
    }
    __syncthreads();
    if (warp == 0) {
      uint32_t wc = (lane < num_warps) ? warp_pairs[lane] : 0;
      uint32_t wt = (lane < num_warps) ? warp_tiles[lane] : 0;
      for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t vc = __shfl_up_sync(0xFFFFFFFFu, wc, o);
        const uint32_t vt = __shfl_up_sync(0xFFFFFFFFu, wt, o);
        if (lane >= o) {
          wc += vc;
          wt += vt;
        }
      }
      if (lane < num_warps) {
        warp_pairs[lane] = wc;
        warp_tiles[lane] = wt;
      }
    }
    __syncthreads();
    const uint32_t excl_c = carry[0] + sc - c + (warp ? warp_pairs[warp - 1] : 0);
    const uint32_t excl_t = carry[1] + st - t + (warp ? warp_tiles[warp - 1] : 0);
    if (b < q.num_buckets) {
      q.bucket_offset[b] = excl_c;
      q.bucket_tile[b] = excl_t;
      q.bucket_cursor[b] = 0;
    }
    __syncthreads();
    if (tid == 0) {
      carry[0] += warp_pairs[num_warps - 1];
      carry[1] += warp_tiles[num_warps - 1];
    }
    __syncthreads();
  }
  if (tid == 0) {
    q.bucket_offset[q.num_buckets] = carry[0];
    q.bucket_tile[q.num_buckets] = carry[1];
    q.counters[SRNXR_QUEUE_COUNTER_PAIRS] = carry[0];
    q.counters[SRNXR_QUEUE_COUNTER_TILES] = carry[1];
  }
}

/* Stage 4: scatter the pairs of request i into the sorted array. */
ccl_device_inline void srnxr_queue_scatter(const SRNXRQueue &q, const uint32_t i)
{
  const uint32_t n = q.req_info[i] & 0xFFu;
  for (uint32_t j = 0; j < n; j++) {
    const uint32_t pair = i * SRNXR_QUEUE_MAX_PAIRS + j;
    const uint32_t bucket = q.pair_bucket[pair];
    const uint32_t pos = q.bucket_offset[bucket] + atomicAdd(q.bucket_cursor + bucket, 1u);
    q.sorted[pos] = pair;
  }
}

/* Bucket of tile t: the last bucket whose first tile is at most t. */
ccl_device_inline uint32_t srnxr_queue_tile_bucket(const SRNXRQueue &q, const uint32_t t)
{
  uint32_t lo = 0, hi = q.num_buckets;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) >> 1;
    if (q.bucket_tile[mid] <= t) {
      lo = mid;
    }
    else {
      hi = mid;
    }
  }
  return lo;
}

#  ifdef SRNXR_HAS_MMA

/* Stage 5: evaluate tile t (16 rows of one network), every lane of the warp calls it. */
template<typename BaseOf>
ccl_device_inline void srnxr_queue_tensor_tile(const SRNXRQueue &q,
                                               const uint32_t t,
                                               BaseOf base_of)
{
  const uint32_t lane = srnxr_lane_id();
  const uint32_t bucket = srnxr_queue_tile_bucket(q, t);
  const uint32_t first = q.bucket_offset[bucket] +
                         (t - q.bucket_tile[bucket]) * SRNXR_QUEUE_TILE_ROWS;
  const uint32_t last = min(first + SRNXR_QUEUE_TILE_ROWS, q.bucket_offset[bucket + 1]);
  const bool has_row = first + lane < last;
  uint32_t pair = 0, leaf = 0, req = 0;
  const ccl_global uint8_t *base = nullptr;
  float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (has_row) {
    pair = q.sorted[first + lane];
    req = pair / SRNXR_QUEUE_MAX_PAIRS;
    leaf = q.pair_leaf[pair];
    base = base_of(q.req_slot[req]);
  }
  /* All rows of the bucket use the same network of equal models (same decode key), take the
   * blob of the first row. */
  const uint32_t leader = __ffs(__ballot_sync(0xFFFFFFFFu, has_row)) - 1;
  const unsigned long long base_bits = __shfl_sync(
      0xFFFFFFFFu, (unsigned long long)(uintptr_t)base, leader);
  const ccl_global uint8_t *tile_base = (const ccl_global uint8_t *)(uintptr_t)base_bits;
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(tile_base);
  const ccl_global SRNXRLeafV2 *leaves = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
      tile_base + hdr->leaves_offset);
  const ccl_global SRNXRNetworkV2 *networks = reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(
      tile_base + hdr->networks_offset);
  uint32_t network = SRNXR_INVALID_INDEX;
  if (has_row) {
    const ccl_global SRNXRLeafV2 &l = leaves[leaf];
    network = l.network;
    const float p[4] = {
        q.req_p[req * 4], q.req_p[req * 4 + 1], q.req_p[req * 4 + 2], q.req_p[req * 4 + 3]};
    srnxr_v2_local_input(l, p, x);
  }
  const uint32_t tile_network = __shfl_sync(0xFFFFFFFFu, network, leader);
  /* A row of another network or model in this bucket (bucket collision) is not evaluated, its
   * request is decoded inline at replay. The blobs of one decode key (the images of one model)
   * hold the same octree and networks, the row may use the blob of the tile. */
  const bool valid = has_row && network == tile_network &&
                     hdr->bucket_base + network == bucket &&
                     (base == tile_base || srnxr_v2_header(base)->decode_key == hdr->decode_key);
  if (has_row && !valid) {
    atomicOr(q.req_info + req, SRNXR_QUEUE_REQUEST_INCOMPLETE);
  }
  const uint32_t rows = __ballot_sync(0xFFFFFFFFu, valid);
  float y[SRNXR_V2_MAX_OUTPUTS];
  const uint32_t width = networks[tile_network].width;
  if (width == 256) {
    srnxr_v2_eval_network_tensor<256>(tile_base, tile_network, rows, x, y);
  }
  else if (width == 128) {
    srnxr_v2_eval_network_tensor<128>(tile_base, tile_network, rows, x, y);
  }
  else if (width == 64) {
    srnxr_v2_eval_network_tensor<64>(tile_base, tile_network, rows, x, y);
  }
  else if (valid) {
    srnxr_v2_eval_network_simt(tile_base, tile_network, x, y);
  }
  if (valid) {
    srnxr_v2_leaf_output(leaves[leaf], y, hdr->output_dim);
    ccl_global float *out = q.pair_out + size_t(pair) * SRNXR_QUEUE_OUT_STRIDE;
    for (uint32_t k = 0; k < hdr->output_dim; k++) {
      out[k] = y[k];
    }
  }
}

/* Stage 5, block-cooperative variant: every warp of the block takes NT consecutive tiles,
 * the block the tiles [first_tile, first_tile + NT warps), and evaluates them bucket by bucket
 * with the weights staged in shared memory (srnxr_v2_eval_network_tensor_block). Every thread of
 * the block calls it; smem: SRNXR_TENSOR_BLOCK_SMEM_BYTES plus 16 bytes. */
template<int NT, typename BaseOf>
ccl_device_inline void srnxr_queue_tensor_group(const SRNXRQueue &q,
                                                const uint32_t first_tile,
                                                const uint32_t num_tiles,
                                                BaseOf base_of,
                                                uint8_t *smem)
{
  const uint32_t lane = srnxr_lane_id();
  const uint32_t warp = threadIdx.x >> 5;
  uint32_t *block_bucket = reinterpret_cast<uint32_t *>(smem + SRNXR_TENSOR_BLOCK_SMEM_BYTES);

  /* The tiles of this warp and the row of this lane in each of them. */
  uint32_t bucket[NT], pair[NT], leaf[NT], req[NT], network[NT];
  bool has_row[NT];
  const ccl_global uint8_t *base[NT];
  float x[4 * NT];
#  pragma unroll
  for (int i = 0; i < NT; i++) {
    const uint32_t t = first_tile + warp * NT + i;
    const bool has_tile = t < num_tiles;
    bucket[i] = has_tile ? srnxr_queue_tile_bucket(q, t) : 0xFFFFFFFFu;
    uint32_t first = 0, last = 0;
    if (has_tile) {
      first = q.bucket_offset[bucket[i]] +
              (t - q.bucket_tile[bucket[i]]) * SRNXR_QUEUE_TILE_ROWS;
      last = min(first + SRNXR_QUEUE_TILE_ROWS, q.bucket_offset[bucket[i] + 1]);
    }
    has_row[i] = has_tile && first + lane < last;
    pair[i] = 0;
    leaf[i] = 0;
    req[i] = 0;
    network[i] = SRNXR_INVALID_INDEX;
    base[i] = nullptr;
    x[4 * i] = x[4 * i + 1] = x[4 * i + 2] = x[4 * i + 3] = 0.0f;
    if (has_row[i]) {
      pair[i] = q.sorted[first + lane];
      req[i] = pair[i] / SRNXR_QUEUE_MAX_PAIRS;
      leaf[i] = q.pair_leaf[pair[i]];
      base[i] = base_of(q.req_slot[req[i]]);
      const ccl_global SRNXRLeafV2 &l = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
          base[i] + srnxr_v2_header(base[i])->leaves_offset)[leaf[i]];
      network[i] = l.network;
      const float p[4] = {q.req_p[req[i] * 4],
                          q.req_p[req[i] * 4 + 1],
                          q.req_p[req[i] * 4 + 2],
                          q.req_p[req[i] * 4 + 3]};
      srnxr_v2_local_input(l, p, x + 4 * i);
    }
  }

  for (;;) {
    /* The lowest bucket still to do in the block. */
    __syncthreads();
    if (threadIdx.x == 0) {
      *block_bucket = 0xFFFFFFFFu;
    }
    __syncthreads();
    if (lane == 0) {
      uint32_t mine = 0xFFFFFFFFu;
#  pragma unroll
      for (int i = 0; i < NT; i++) {
        mine = min(mine, bucket[i]);
      }
      if (mine != 0xFFFFFFFFu) {
        atomicMin(block_bucket, mine);
      }
    }
    __syncthreads();
    const uint32_t cur = *block_bucket;
    if (cur == 0xFFFFFFFFu) {
      break;
    }
    /* Model and network of the bucket from its first pair, uniform over the block. */
    const uint32_t pair0 = q.sorted[q.bucket_offset[cur]];
    const ccl_global uint8_t *tile_base = base_of(q.req_slot[pair0 / SRNXR_QUEUE_MAX_PAIRS]);
    const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(tile_base);
    const ccl_global SRNXRLeafV2 *leaves = reinterpret_cast<const ccl_global SRNXRLeafV2 *>(
        tile_base + hdr->leaves_offset);
    const ccl_global SRNXRNetworkV2 *networks =
        reinterpret_cast<const ccl_global SRNXRNetworkV2 *>(tile_base + hdr->networks_offset);
    const uint32_t tile_network = leaves[q.pair_leaf[pair0]].network;
    /* A row of another network or model in this bucket (bucket collision) is not evaluated, its
     * request is decoded inline at replay. The blobs of one decode key (the images of one model)
     * hold the same octree and networks, the row may use the blob of the tile. */
    uint32_t rows[NT];
    bool valid[NT];
#  pragma unroll
    for (int i = 0; i < NT; i++) {
      const bool mine = bucket[i] == cur;
      valid[i] = mine && has_row[i] && network[i] == tile_network &&
                 hdr->bucket_base + network[i] == cur &&
                 (base[i] == tile_base ||
                  srnxr_v2_header(base[i])->decode_key == hdr->decode_key);
      if (mine && has_row[i] && !valid[i]) {
        atomicOr(q.req_info + req[i], SRNXR_QUEUE_REQUEST_INCOMPLETE);
      }
      rows[i] = __ballot_sync(0xFFFFFFFFu, valid[i]);
    }
    float y[SRNXR_V2_MAX_OUTPUTS * NT];
    const uint32_t width = networks[tile_network].width;
    const bool split = (hdr->flags & SRNXR_DEVICE_FLAG_TENSOR_SPLIT) != 0;
    bool done = true;
    if (width == 256) {
      if (split) {
        srnxr_v2_eval_network_tensor_block<256, true, NT>(tile_base, tile_network, rows, x, y, smem);
      }
      else {
        srnxr_v2_eval_network_tensor_block<256, false, NT>(tile_base, tile_network, rows, x, y, smem);
      }
    }
    else if (width == 128) {
      if (split) {
        srnxr_v2_eval_network_tensor_block<128, true, NT>(tile_base, tile_network, rows, x, y, smem);
      }
      else {
        srnxr_v2_eval_network_tensor_block<128, false, NT>(tile_base, tile_network, rows, x, y, smem);
      }
    }
    else if (width == 64) {
      if (split) {
        srnxr_v2_eval_network_tensor_block<64, true, NT>(tile_base, tile_network, rows, x, y, smem);
      }
      else {
        srnxr_v2_eval_network_tensor_block<64, false, NT>(tile_base, tile_network, rows, x, y, smem);
      }
    }
    else {
      done = false;
    }
#  pragma unroll
    for (int i = 0; i < NT; i++) {
      if (!valid[i]) {
        continue;
      }
      if (!done) {
        srnxr_v2_eval_network_simt(tile_base, tile_network, x + 4 * i, y + SRNXR_V2_MAX_OUTPUTS * i);
      }
      srnxr_v2_leaf_output(leaves[leaf[i]], y + SRNXR_V2_MAX_OUTPUTS * i, hdr->output_dim);
      ccl_global float *out = q.pair_out + size_t(pair[i]) * SRNXR_QUEUE_OUT_STRIDE;
      for (uint32_t k = 0; k < hdr->output_dim; k++) {
        out[k] = y[SRNXR_V2_MAX_OUTPUTS * i + k];
      }
    }
#  pragma unroll
    for (int i = 0; i < NT; i++) {
      if (bucket[i] == cur) {
        bucket[i] = 0xFFFFFFFFu;
      }
    }
  }
}

#  endif /* SRNXR_HAS_MMA */

/* Stage 6: blend the pairs of request i in neighbor order. */
template<typename BaseOf>
ccl_device_inline void srnxr_queue_blend(const SRNXRQueue &q, const uint32_t i, BaseOf base_of)
{
  const uint32_t info = q.req_info[i];
  const ccl_global uint8_t *base = base_of(q.req_slot[i]);
  const ccl_global SRNXRDeviceHeaderV2 *hdr = srnxr_v2_header(base);
  const uint32_t output_dim = hdr->output_dim;
  ccl_global float *out = q.req_out + size_t(i) * SRNXR_QUEUE_OUT_STRIDE;
  if (info & SRNXR_QUEUE_REQUEST_INCOMPLETE) {
    return;
  }
  const uint32_t n = info & 0xFFu;
  if (n == 0) {
    for (uint32_t k = 0; k < output_dim; k++) {
      out[k] = hdr->outside_value;
    }
    return;
  }
  SRNXRBlend blend;
  srnxr_blend_init(blend);
  for (uint32_t j = 0; j < n; j++) {
    const uint32_t pair = i * SRNXR_QUEUE_MAX_PAIRS + j;
    float u[SRNXR_V2_MAX_OUTPUTS];
    const ccl_global float *po = q.pair_out + size_t(pair) * SRNXR_QUEUE_OUT_STRIDE;
    for (uint32_t k = 0; k < output_dim; k++) {
      u[k] = po[k];
    }
    srnxr_blend_add(blend, q.pair_exp[pair], u, output_dim);
  }
  const float inv = srnxr_div(1.0f, blend.weight_sum);
  for (uint32_t k = 0; k < output_dim; k++) {
    out[k] = blend.value[k] * inv;
  }
}

/* Add the per launch counters to the statistics (one thread). */
ccl_device_inline void srnxr_queue_finish_stats(const SRNXRQueue &q)
{
  atomicAdd(q.stats + SRNXR_QUEUE_COUNTER_REQUESTS,
            (unsigned long long int)srnxr_queue_num_requests(q));
  atomicAdd(q.stats + SRNXR_QUEUE_COUNTER_PAIRS,
            (unsigned long long int)q.counters[SRNXR_QUEUE_COUNTER_PAIRS]);
  atomicAdd(q.stats + SRNXR_QUEUE_COUNTER_TILES,
            (unsigned long long int)q.counters[SRNXR_QUEUE_COUNTER_TILES]);
  atomicAdd(q.stats + SRNXR_QUEUE_COUNTER_LAUNCHES, 1ull);
}

#endif /* __KERNEL_CUDA__ && __CUDA_ARCH__ */

CCL_NAMESPACE_END
