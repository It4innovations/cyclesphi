/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Shared by the Cycles host code, the Cycles kernels and the standalone SRNXR tools, which
 * define SRNXR_STANDALONE and provide uint32_t/uint64_t and CCL_NAMESPACE_BEGIN themselves. */
#ifndef SRNXR_STANDALONE
#  include "util/types.h"
#endif

CCL_NAMESPACE_BEGIN

// ============================================================================
// SRNXR (SIREN octree) Volume Format - On-Device Structures
// ============================================================================
// SRNXRImageLoader parses and validates the .srnxr file on the host and uploads
// a compact blob whose first two words are the magic and the device version.
//
// Device version 1 (file format 1): 3D blocks with one output, uniform CSR grid.
//   1. SRNXRDeviceHeader (aligned to SRNXR_DEVICE_ALIGNMENT)
//   2. float    block_centers[block_count * 3]   (in shifted coordinates)
//   3. float    block_scales[block_count * 3]    (half-extents per axis)
//   4. float    block_stats[block_count * 2]     (mu, sigma)
//   5. uint32_t block_layout[block_count * 4]    (width, depth, weight_offset, bias_offset)
//   6. float    siren_weights[]                  (FP32, converted from FP16 if needed)
//   7. float    siren_biases[]
//   8. uint32_t grid_offsets[cell_count + 1]     (CSR row pointers)
//   9. uint32_t grid_indices[]                   (CSR candidate block indices)
//
// Device version 2 (file format 2): adaptive octree of leaves with local SIREN
// networks over (x, y, z, t) and several output fields, see
// tools/srnxr/neural_volume.md for the decoding contract.
//   1. SRNXRDeviceHeaderV2
//   2. SRNXRNodeV2    nodes[node_count]
//   3. SRNXRLeafV2    leaves[leaf_count]
//   4. uint32_t       neighbors[]                (leaves whose expanded box touches a leaf)
//   5. SRNXRNetworkV2 networks[network_count]
//   6. uint16_t       weights[]                  (FP16, row-major [out][in] per layer)
//   7. float          biases[]                   (FP32)
//   8. uint32_t       mma_weights[]              (FP16 pairs in mma.m16n8k16 B fragment order)
//   8b. uint16_t      warp_weights[]             (FP16, hidden layers transposed [in][out])
//   9. uint64_t       counters[SRNXR_COUNTER_NUM] (device statistics, zero on upload)
// Every array starts at a multiple of SRNXR_DEVICE_ALIGNMENT, offsets are in
// bytes from the start of the blob.
// ============================================================================

#define SRNXR_DEVICE_MAGIC 0x584E5253u /* 'SRNX' */
#define SRNXR_DEVICE_VERSION 1u
#define SRNXR_DEVICE_VERSION_V2 2u
#define SRNXR_DEVICE_ALIGNMENT 64u

/* Maximum block width the kernel scratch buffers accommodate, the loader rejects wider files. */
#define SRNXR_MAX_WIDTH 64

/* BLOCK_LAYOUT and BLOCK_STATS row sizes. */
#define SRNXR_BLOCK_LAYOUT_FIELDS 4
#define SRNXR_BLOCK_STATS_FIELDS 2

struct SRNXRDeviceHeader {
  uint32_t magic;               // SRNXR_DEVICE_MAGIC
  uint32_t version;             // SRNXR_DEVICE_VERSION
  uint32_t block_count;         // Number of blocks
  uint32_t hidden_layer_count;  // Number of sine layers per block

  uint32_t max_width;           // Largest block width
  uint32_t grid_dim_x;          // Acceleration grid dimensions
  uint32_t grid_dim_y;
  uint32_t grid_dim_z;

  float domain_min_x;           // Domain bounds in object space
  float domain_min_y;
  float domain_min_z;
  float domain_max_x;
  float domain_max_y;
  float domain_max_z;

  float w0;                     // SIREN frequency factor
  float overlap;                // Block membership radius in normalized block units
  float blend_sigma;            // Width of the blending weight falloff
  float weight_sum_epsilon;     // Minimum blended weight sum, zero density below

  float shift_x;                // Off-grid modulo shift (normalized domain units)
  float shift_y;
  float shift_z;
  float value_mean;             // Global denormalization
  float value_std;
  uint32_t reserved0;

  uint64_t block_centers_offset;
  uint64_t block_scales_offset;
  uint64_t block_stats_offset;
  uint64_t block_layout_offset;
  uint64_t siren_weights_offset;
  uint64_t siren_biases_offset;
  uint64_t grid_offsets_offset;
  uint64_t grid_indices_offset;
  uint64_t total_size;          // Total blob size in bytes
};

// ----------------------------------------------------------------------------
// Device version 2
// ----------------------------------------------------------------------------

/* Limits of the version 2 decoders, the loader rejects files exceeding them. */
#define SRNXR_V2_MAX_OUTPUTS 8
#define SRNXR_V2_MAX_LAYERS 8
#define SRNXR_V2_MAX_WIDTH 256
#define SRNXR_V2_MAX_CHANNELS 3
#define SRNXR_INVALID_INDEX 0xFFFFFFFFu

/* Inference backends, the logical configuration values of the decoder. The kernels never
 * substitute one for another silently: a backend that cannot run a query counts it in
 * SRNXR_COUNTER_FALLBACK_SIMT and evaluates it with the CUDA_FUSED arithmetic. */
enum SRNXRBackend {
  /* FP16 stored weights, FP32 multiply-accumulate and FP32 sine, one query per thread. */
  SRNXR_BACKEND_CUDA_FUSED = 0,
  /* FP16 weights and activations on mma.sync.m16n8k16 Tensor Cores with FP32 accumulation,
   * FP32 bias, sine and first layer. */
  SRNXR_BACKEND_CUDA_TENSOR = 1,
  /* OptiX cooperative vectors, FP16 matrices and vectors, only inside OptiX programs. */
  SRNXR_BACKEND_COOPVEC = 2,
  SRNXR_BACKEND_NUM = 3,
};

enum SRNXRSineMode {
  SRNXR_SINE_ACCURATE = 0, /* sinf */
  SRNXR_SINE_FAST = 1,     /* __sinf on CUDA, sinf elsewhere */
};

/* Flags of SRNXRDeviceHeaderV2. */
enum SRNXRDeviceFlags {
  SRNXR_DEVICE_FLAG_MMA_WEIGHTS = (1 << 0), /* mma_weights holds every network */
  SRNXR_DEVICE_FLAG_COUNTERS = (1 << 1),    /* the kernels update the counters */
  SRNXR_DEVICE_FLAG_WARP_WEIGHTS = (1 << 2), /* warp_weights holds the networks of width 64k */
  SRNXR_DEVICE_FLAG_PARTIAL_WARPS = (1 << 3), /* use the partial-warp path (else per-thread) */
  /* The majorant bake samples the decoded field like any procedural density instead of
   * returning the certified bounds: tight, but NOT certified (no guarantee between samples). */
  SRNXR_DEVICE_FLAG_SAMPLED_MAJORANT = (1 << 4),
  /* Wavefront queue Tensor Core stage: split every FP16 activation into a high and a low
   * FP16 part (two mma streams), FP32-level activations. */
  SRNXR_DEVICE_FLAG_TENSOR_SPLIT = (1 << 5),
};

/* Device statistics, see srnxr_counter_add(). */
enum SRNXRCounter {
  SRNXR_COUNTER_QUERIES = 0,         /* calls inside the domain */
  SRNXR_COUNTER_OUTSIDE,             /* calls outside the domain */
  SRNXR_COUNTER_NETWORK_EVALS,       /* local network evaluations */
  SRNXR_COUNTER_SIMT_EVALS,          /* network evaluations by the per-thread SIMT path */
  SRNXR_COUNTER_TENSOR_TILES,        /* mma tiles evaluated */
  SRNXR_COUNTER_TENSOR_ROWS,         /* valid rows in those tiles */
  SRNXR_COUNTER_FALLBACK_SIMT,       /* evaluations a non-SIMT backend had to run as SIMT */
  SRNXR_COUNTER_COOPVEC_EVALS,       /* network evaluations by cooperative vectors */
  SRNXR_COUNTER_BAKE_QUERIES,        /* conservative bound queries of the majorant bake */
  SRNXR_COUNTER_WARP_SIMT_EVALS,     /* network evaluations by the warp-cooperative SIMT path */
  SRNXR_COUNTER_WARP_SIMT_PASSES,    /* row groups of the warp-cooperative SIMT path */
  SRNXR_COUNTER_PARTIAL_WARP_EVALS,  /* network evaluations by the partial-warp SIMT path */
  SRNXR_COUNTER_WARP_CALLS,          /* warps entering the warp-cooperative decoder */
  SRNXR_COUNTER_WARP_ACTIVE_LANES,   /* active lanes summed over those calls */
  SRNXR_COUNTER_NUM = 16,
};

/* Number of SRNXR images whose statistics the Cycles CUDA kernels keep (srnxr_counters). */
#define SRNXR_COUNTER_SLOTS 64

#if !defined(__KERNEL_GPU__) || defined(SRNXR_STANDALONE)
inline const char *srnxr_counter_name(const int counter)
{
  static const char *names[SRNXR_COUNTER_NUM] = {"queries",
                                                 "outside",
                                                 "network_evals",
                                                 "simt_evals",
                                                 "tensor_tiles",
                                                 "tensor_rows",
                                                 "fallback_simt",
                                                 "coopvec_evals",
                                                 "bake_queries",
                                                 "warp_simt_evals",
                                                 "warp_simt_passes",
                                                 "partial_warp_evals",
                                                 "warp_calls",
                                                 "warp_active_lanes",
                                                 "unused14",
                                                 "unused15"};
  return (counter >= 0 && counter < SRNXR_COUNTER_NUM) ? names[counter] : "unused";
}
#endif

struct SRNXRNodeV2 {
  float bmin[3];
  uint32_t first_child; /* index of the 8 children, SRNXR_INVALID_INDEX for a leaf node */
  float bmax[3];
  uint32_t leaf;        /* leaf index of a leaf node */
};

struct SRNXRLeafV2 {
  float expanded_min[4]; /* support of the leaf network, closed box in (x, y, z, t) */
  float expanded_max[4];
  float input_offset[4]; /* x_local = (p - input_offset) * input_scale */
  float input_scale[4];
  float blend_center[4]; /* r^2 = sum(((p - blend_center) * blend_scale)^2) */
  float blend_scale[4];
  float blend_inv_two_sigma_sq; /* e = -r^2 * blend_inv_two_sigma_sq */
  uint32_t network;
  uint32_t neighbor_begin; /* neighbors[begin, end): leaves to test for a point in this leaf */
  uint32_t neighbor_end;
  float output_scale[SRNXR_V2_MAX_OUTPUTS]; /* u = y * output_scale + output_offset */
  float output_offset[SRNXR_V2_MAX_OUTPUTS];
  /* Certified range of the mapped channels of the output of the leaf network, for any input
   * (|sin| <= 1). */
  float bound_min[SRNXR_V2_MAX_CHANNELS];
  float bound_max[SRNXR_V2_MAX_CHANNELS];
  float pad[2];
};

struct SRNXRNetworkV2 {
  uint32_t width;
  uint32_t weight_offset; /* elements into weights[] */
  uint32_t bias_offset;   /* elements into biases[] */
  uint32_t mma_offset;    /* uint32 elements into mma_weights[], SRNXR_INVALID_INDEX if none */
  /* uint16 elements into warp_weights[] (transposed hidden layers for the warp-cooperative
   * SIMT path), SRNXR_INVALID_INDEX if none. */
  uint32_t warp_offset;
  /* Bytes into the cooperative vector region, SRNXR_INVALID_INDEX if none. The block of a
   * network: hidden layer matrices 1 .. layers - 1 (W x W), the output matrix (8 x W, rows
   * beyond output_dim zero), FP32 biases ((layers - 1) W + 8), FP16 biases (same count), every
   * part 64-byte aligned. */
  uint32_t coopvec_offset;
  uint32_t pad[2];
};

/* A cooperative vector matrix to convert from row-major FP16 (at staging_offset) into the
 * INFERENCING_OPTIMAL layout (at blob_offset, size bytes). Offsets in bytes from the start of the
 * blob. */
struct SRNXRCoopVecMatrixV2 {
  uint32_t n;
  uint32_t k;
  uint64_t staging_offset;
  uint64_t blob_offset;
  uint64_t size;
};

struct SRNXRDeviceHeaderV2 {
  uint32_t magic;   /* SRNXR_DEVICE_MAGIC */
  uint32_t version; /* SRNXR_DEVICE_VERSION_V2 */
  uint32_t input_dim;
  uint32_t output_dim;

  uint32_t hidden_layer_count;
  uint32_t hidden_width;
  uint32_t network_count;
  uint32_t leaf_count;

  uint32_t node_count;
  uint32_t flags;     /* SRNXRDeviceFlags */
  uint32_t backend;   /* SRNXRBackend */
  uint32_t sine_mode; /* SRNXRSineMode */

  /* The image returns (u[channel_field[0]], u[channel_field[1]], u[channel_field[2]], 1), a
   * single channel is replicated to xyz. */
  uint32_t channel_count;
  uint32_t channel_field[SRNXR_V2_MAX_CHANNELS];

  float domain_min[4];
  float domain_max[4];

  /* Simulation time of a sample: time_frame + time_shutter * (ray time - 0.5). */
  float time_frame;
  float time_shutter;
  /* Half size of the box around a majorant bake sample that is bounded conservatively. */
  float bake_margin[3];
  float outside_value; /* value of every channel outside the domain */
  /* Slot of the statistics in the Cycles kernel counters (srnxr_counters), set on upload. */
  uint32_t counter_slot;
  /* COOPVEC: 0 matrix products with FP32 output and bias, 1 with FP16 output and bias. */
  uint32_t coopvec_mode;
  /* Nonzero key of the decoded field (file checksum and the decode settings except the channel
   * mapping): images with the same key share the decode of a point (ShaderData cache). */
  uint32_t decode_key;
  /* First bucket of the networks of this model in the wavefront decode queue
   * (kernel/util/image_3d_srnxr_queue.h), set on upload; SRNXR_QUEUE_INVALID_BUCKET_BASE
   * (0xFFFFFFFF) when the image is not decoded through the queue. */
  uint32_t bucket_base;

  float layer_w0[SRNXR_V2_MAX_LAYERS];

  uint64_t nodes_offset;
  uint64_t leaves_offset;
  uint64_t neighbors_offset;
  uint64_t networks_offset;
  uint64_t weights_offset;
  uint64_t biases_offset;
  uint64_t mma_weights_offset;
  uint64_t warp_weights_offset;
  /* OptiX cooperative vector matrices (INFERENCING_OPTIMAL layout) and biases, filled on the
   * device by optixCoopVecMatrixConvert(), 0 when absent. */
  uint64_t coopvec_offset;
  uint64_t coopvec_size;
  /* Row-major FP16 source of the conversion and the list of matrices (SRNXRCoopVecMatrixV2),
   * read by OptiXDevice when it uploads the blob. */
  uint64_t coopvec_staging_offset;
  uint64_t coopvec_table_offset;
  uint64_t coopvec_matrix_count;
  uint64_t counters_offset;
  uint64_t total_size;
};

CCL_NAMESPACE_END
