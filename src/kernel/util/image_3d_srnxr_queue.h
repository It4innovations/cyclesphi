/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Wavefront ("queued") decoding of SRNXR format 2 volumes.
 *
 * The Cycles volume integrator walks every ray in one thread, decoding the field at every
 * candidate collision; the launch lasts as long as the longest walk, the warps are partially
 * active at the decoder and the Tensor Core path gets 1 - 16 rows per tile. The queue takes the
 * decodes out of the walk:
 *
 *   1. record: the walk runs with the decoder replaced by "append the query to the window of
 *      this launch slot, return zero". The candidate positions of null-collision tracking depend
 *      on the majorant and the random numbers only, so the recorded set is a superset of the
 *      queries of the real walk (which may stop earlier: Russian roulette, zero throughput).
 *   2. expand: every request is resolved to its leaves (the strict leaf and the neighbors whose
 *      expanded box contains the point) with the blend exponent, one pair per leaf, counted per
 *      bucket (model, network).
 *   3. scan: pair and tile offsets of the buckets (counting sort).
 *   4. scatter: pair indices sorted by bucket.
 *   5. tensor: one warp per tile of 16 rows of one network (mma.m16n8k16), consecutive tiles of
 *      one network in one block so the weight fragments are reused from L1.
 *   6. blend: the pairs of a request are blended in neighbor order (deterministic) into the
 *      result of the request.
 *   7. replay: the real walk looks every query up in its window (exact position, model) and
 *      decodes inline when it is not there (counted).
 *
 * Shared by the Cycles CUDA kernels and the standalone bench (tools/srnxr/bench), which provides
 * the SRNXR_STANDALONE definitions. This header holds the structures (IntegratorStateGPU and the
 * host refer to them), kernel/util/image_3d_srnxr_queue_decode.h the CUDA device code of the
 * stages, which needs the decoder (kernel/util/image_3d_srnxr_decode.h). */

CCL_NAMESPACE_BEGIN

/* Leaves a request can blend through the queue; requests with more candidates are decoded
 * inline at replay (counted as incomplete). */
#define SRNXR_QUEUE_MAX_PAIRS 8
/* Rows of a Tensor Core tile. */
#define SRNXR_QUEUE_TILE_ROWS 16
/* Floats per result. */
#define SRNXR_QUEUE_OUT_STRIDE 8 /* SRNXR_V2_MAX_OUTPUTS */
/* Default lookups per launch slot (window). */
#define SRNXR_QUEUE_DEFAULT_K_MAX 128
#define SRNXR_QUEUE_DEFAULT_BUCKETS 65536
#define SRNXR_QUEUE_INVALID_BUCKET_BASE 0xFFFFFFFFu

enum SRNXRQueueMode {
  SRNXR_QUEUE_OFF = 0,
  SRNXR_QUEUE_RECORD = 1,
  SRNXR_QUEUE_REPLAY = 2,
};

/* Per launch counters (zeroed before the record kernel) and cumulative statistics. */
enum SRNXRQueueCounter {
  SRNXR_QUEUE_COUNTER_REQUESTS = 0, /* requests recorded (may exceed capacity, clamp) */
  SRNXR_QUEUE_COUNTER_PAIRS,        /* leaf evaluations of the batch */
  SRNXR_QUEUE_COUNTER_TILES,        /* Tensor Core tiles of the batch */
  SRNXR_QUEUE_COUNTER_OVERFLOW,     /* lookups not recorded (window or capacity full) */
  SRNXR_QUEUE_COUNTER_INCOMPLETE,   /* requests with more candidates than MAX_PAIRS */
  SRNXR_QUEUE_COUNTER_REPLAY_HITS,  /* lookups served from the queue */
  SRNXR_QUEUE_COUNTER_REPLAY_MISS,  /* lookups decoded inline at replay */
  SRNXR_QUEUE_COUNTER_LAUNCHES,     /* record launches */
  SRNXR_QUEUE_COUNTER_NUM = 8,
};

ccl_device_inline const char *srnxr_queue_counter_name(const int counter)
{
  static const char *names[SRNXR_QUEUE_COUNTER_NUM] = {"requests",
                                                       "pairs",
                                                       "tiles",
                                                       "overflow",
                                                       "incomplete",
                                                       "replay_hits",
                                                       "replay_miss",
                                                       "launches"};
  return (counter >= 0 && counter < SRNXR_QUEUE_COUNTER_NUM) ? names[counter] : "unused";
}

/* req_info flags (bits 8+), bits 0 - 7 hold the pair count. */
enum SRNXRQueueRequestFlag {
  SRNXR_QUEUE_REQUEST_INCOMPLETE = (1u << 8), /* decode inline at replay */
  SRNXR_QUEUE_REQUEST_OUTSIDE = (1u << 9),    /* outside the domain, result = outside value */
};

struct SRNXRQueue {
  /* Requests. */
  ccl_global float *req_p;        /* capacity x 4: (x, y, z, simulation time) */
  ccl_global uint32_t *req_slot;  /* capacity: image info id of the recording image */
  ccl_global uint32_t *req_key;   /* capacity: decode key of the model */
  ccl_global uint32_t *req_info;  /* capacity: pair count | SRNXRQueueRequestFlag */
  ccl_global float *req_out;      /* capacity x SRNXR_QUEUE_OUT_STRIDE */
  /* Pairs (request, leaf), SRNXR_QUEUE_MAX_PAIRS per request in neighbor order. */
  ccl_global uint32_t *pair_leaf;   /* capacity x MAX_PAIRS */
  ccl_global uint32_t *pair_bucket; /* capacity x MAX_PAIRS */
  ccl_global float *pair_exp;       /* capacity x MAX_PAIRS: blend exponent */
  ccl_global float *pair_out;       /* capacity x MAX_PAIRS x OUT_STRIDE: mapped network output */
  ccl_global uint32_t *sorted;      /* capacity x MAX_PAIRS: pair indices sorted by bucket */
  /* Buckets (model, network): bucket_base of the model + network index. */
  ccl_global uint32_t *bucket_count;  /* num_buckets */
  ccl_global uint32_t *bucket_offset; /* num_buckets + 1: first pair */
  ccl_global uint32_t *bucket_tile;   /* num_buckets + 1: first tile */
  ccl_global uint32_t *bucket_cursor; /* num_buckets: scatter cursor */
  /* Windows of the launch slots: window[k * num_slots + slot] is the request of lookup k. */
  ccl_global uint32_t *window;
  ccl_global uint32_t *window_count; /* num_slots */
  /* Per launch counters (SRNXRQueueCounter) and cumulative statistics. */
  ccl_global uint32_t *counters;
  ccl_global unsigned long long int *stats;
  uint32_t capacity;
  uint32_t num_buckets;
  uint32_t k_max;
  uint32_t num_slots;
};

CCL_NAMESPACE_END
