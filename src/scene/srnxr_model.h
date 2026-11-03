/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Host side of the .srnxr format version 2: parsing, validation and packing into the device
 * blob of kernel/util/image_3d_srnxr.h. Only depends on the standard library so the
 * standalone SRNXR tools (tools/srnxr) build the very same blob as SRNXRImageLoader. */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#ifdef SRNXR_STANDALONE
#  ifndef CCL_NAMESPACE_BEGIN
#    define CCL_NAMESPACE_BEGIN namespace ccl {
#    define CCL_NAMESPACE_END }
#  endif
#endif

#include "kernel/util/image_3d_srnxr.h"

CCL_NAMESPACE_BEGIN

/* Per-image decoder configuration, it is part of the device blob. */
struct SRNXRBuildSettings {
  SRNXRBackend backend = SRNXR_BACKEND_CUDA_FUSED;
  SRNXRSineMode sine_mode = SRNXR_SINE_ACCURATE;
  /* Names of the fields returned in the image channels: one (replicated to xyz) or three.
   * Empty selects the only field, or the field named "rho" or "density". */
  std::vector<std::string> fields;
  /* Simulation time of the frame, required for 4D models unless the domain time range is a
   * single value; time_shutter maps the ray time to [-0.5, 0.5] * time_shutter around it. */
  bool has_time_frame = false;
  float time_frame = 0.0f;
  float time_shutter = 0.0f;
  float outside_value = 0.0f;
  /* Depth of the Cycles majorant grid (VOLUME_OCTREE_MAX_DEPTH), sets the bake margin. */
  int bake_grid_depth = 7;
  /* Pack the mma.m16n8k16 weight copy even when the backend is not CUDA_TENSOR. */
  bool force_mma_weights = false;
  /* Pack the transposed copy of the warp-cooperative SIMT path for CUDA_FUSED (GPU only, the
   * CPU runs one query per thread). */
  bool warp_weights = true;
  /* Pack it even when the backend is not CUDA_FUSED. */
  bool force_warp_weights = false;
  /* Update the device statistics (SRNXRCounter). */
  bool counters = true;
  /* Decode partially active warps cooperatively (otherwise one query per thread). */
  bool partial_warps = true;
  /* Majorant bake from the decoded field at the bake samples (SRNXR_DEVICE_FLAG_SAMPLED_MAJORANT,
   * not certified) instead of the certified bounds. */
  bool sampled_majorant = false;
  /* COOPVEC: size in bytes of an N x K FP16 matrix in the OptiX INFERENCING_OPTIMAL layout
   * (optixCoopVecMatrixComputeSize). Without it no cooperative vector region is packed. */
  std::function<uint64_t(uint32_t n, uint32_t k)> coopvec_matrix_size;
  /* COOPVEC: 1 FP16 products and biases, 0 FP32 (rejected by OptiX 9.1 for FP16 inputs, only
   * built with SRNXR_COOPVEC_TRY_FP32_OUTPUT), see SRNXRDeviceHeaderV2::coopvec_mode. */
  uint32_t coopvec_mode = 1;
};

/* A matrix the caller converts with optixCoopVecMatrixConvert() from the row-major FP16
 * staging buffer of the report into the blob, after uploading the blob. */
struct SRNXRCoopVecMatrix {
  uint32_t n;
  uint32_t k;
  uint64_t staging_offset; /* bytes into SRNXRBuildReport::coopvec_staging, 64 aligned */
  uint64_t blob_offset;    /* bytes into the device blob, 64 aligned */
  uint64_t size;           /* bytes of the converted matrix */
};

struct SRNXRBuildReport {
  uint32_t format_version = 0;
  uint32_t input_dim = 0;
  uint32_t output_dim = 0;
  uint32_t hidden_layer_count = 0;
  uint32_t hidden_width = 0;
  uint32_t network_count = 0;
  uint32_t leaf_count = 0;
  uint32_t node_count = 0;
  uint32_t max_depth = 0;
  std::vector<std::string> field_names;
  std::vector<float> layer_w0;
  std::vector<uint32_t> channel_field;
  bool synthetic = false;
  bool weights_were_fp32 = false;
  bool has_fp32_master = false;

  /* Neighborhood statistics */
  uint32_t max_neighbors = 0;
  double mean_neighbors = 0.0;
  /* Expansion of the leaves beyond their strict box, relative to the strict size, over all
   * leaves, axes and both sides; sides clipped by the domain boundary are excluded. */
  double overlap_min = 0.0;
  double overlap_max = 0.0;
  uint32_t networks_without_mma = 0;
  uint32_t networks_without_warp = 0;

  size_t file_bytes = 0;
  size_t device_bytes = 0;
  size_t weight_bytes = 0;
  size_t mma_weight_bytes = 0;
  size_t warp_weight_bytes = 0;
  size_t coopvec_bytes = 0;
  std::vector<uint16_t> coopvec_staging;
  std::vector<SRNXRCoopVecMatrix> coopvec_matrices;
  double parse_seconds = 0.0;
  double pack_seconds = 0.0;

  std::vector<std::string> warnings;
  std::string metadata_json;

  std::string summary() const;
};

/* True when data starts with the .srnxr magic and declares format version 2. */
bool srnxr_is_version2(const char *data, size_t size);

/* Parse, validate and pack a version 2 file. On failure the blob is empty and error says why. */
bool srnxr_build_device_blob_v2(const char *data,
                                size_t size,
                                const SRNXRBuildSettings &settings,
                                std::vector<char> &blob,
                                SRNXRBuildReport &report,
                                std::string &error);

/* Names used on the command line, in XML and in logs. */
const char *srnxr_backend_name(SRNXRBackend backend);
bool srnxr_backend_from_name(const std::string &name, SRNXRBackend &backend);
const char *srnxr_sine_mode_name(SRNXRSineMode mode);
bool srnxr_sine_mode_from_name(const std::string &name, SRNXRSineMode &mode);

CCL_NAMESPACE_END
