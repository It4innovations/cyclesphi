/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

// ============================================================================
// SRNXR (SIREN octree) Volume Format - On-Device Structures
// ============================================================================
// The .srnxr file stores a volume as a set of overlapping blocks, each with
// its own small sine-activated MLP (SIREN), plus a uniform world-space CSR
// acceleration grid listing the candidate blocks of every cell.
//
// SRNXRImageLoader parses and validates the file on the host and uploads a
// compact blob, laid out as:
//   1. SRNXRDeviceHeader (aligned to SRNXR_DEVICE_ALIGNMENT)
//   2. float    block_centers[block_count * 3]   (in shifted coordinates)
//   3. float    block_scales[block_count * 3]    (half-extents per axis)
//   4. float    block_stats[block_count * 2]     (mu, sigma)
//   5. uint32_t block_layout[block_count * 4]    (width, depth, weight_offset, bias_offset)
//   6. float    siren_weights[]                  (FP32, converted from FP16 if needed)
//   7. float    siren_biases[]
//   8. uint32_t grid_offsets[cell_count + 1]     (CSR row pointers)
//   9. uint32_t grid_indices[]                   (CSR candidate block indices)
// Every array starts at a multiple of SRNXR_DEVICE_ALIGNMENT, offsets are in
// bytes from the start of the blob.
// ============================================================================

#define SRNXR_DEVICE_MAGIC 0x584E5253u /* 'SRNX' */
#define SRNXR_DEVICE_VERSION 1u
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

CCL_NAMESPACE_END
