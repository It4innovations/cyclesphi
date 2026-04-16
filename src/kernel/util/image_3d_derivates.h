/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

// ============================================================================
// NanoVDB Derivative Bundle Format - On-Device Structures
// ============================================================================
// Binary layout:
//   1. DerivFileHeader (64 bytes, aligned to 32)
//   2. DerivLevelHeader[levelCount] (16 bytes each)
//   3. DerivGridHeader[gridCount] (128 bytes each)
//   4. NanoVDB grid payloads (each 32-byte aligned)
//
// This format supports two modes:
//   - Single derivative per grid (IMAGE_DATA_TYPE_NANOVDB_DERIVATES)
//   - Packed derivatives (IMAGE_DATA_TYPE_NANOVDB_DERIVATES_VEC4)
//     Up to 4 derivatives packed per grid using gridType field
// ============================================================================

// Grid type for packed derivative format
enum DerivGridType : uint32_t {
  DERIV_GRID_TYPE_FLOAT = 0,   // Single float (1 derivative)
  DERIV_GRID_TYPE_VEC3F = 1,   // Vec3f (2-3 derivatives packed)
  DERIV_GRID_TYPE_VEC4F = 2,   // Vec4f (4 derivatives packed)
};

struct DerivFileHeader {
  uint32_t magic;              // Magic number: 0x4E56444D ('NVDM')
  uint32_t version;            // File format version
  uint32_t payloadAlignment;   // Alignment requirement for grid payloads
  uint32_t levelCount;         // Number of levels
  uint32_t gridCount;          // Total number of grids
  uint32_t reserved1;
  uint64_t levelTableOffset;   // Byte offset to LevelHeader array
  uint64_t gridTableOffset;    // Byte offset to GridHeader array
  uint64_t payloadBlockOffset; // Byte offset to first grid payload
  uint64_t totalFileSize;      // Total file size in bytes
  uint64_t reserved2;
};

struct DerivLevelHeader {
  uint32_t levelIndex;         // Level index (0-based)
  uint32_t derivativeCount;    // Number of derivatives in this level
  uint32_t firstGridIndex;     // Index of first grid in GridHeader array
  uint32_t reserved;
};

struct DerivGridHeader {
  uint32_t levelIndex;                // Which level this grid belongs to
  uint32_t derivativeIndex;           // Derivative index within level (0-based)
  uint32_t derivativeCountInLevel;    // Total derivatives in this level
  uint32_t gridType;                  // Grid type: DERIV_GRID_TYPE_FLOAT/VEC3F/VEC4F
  uint64_t payloadOffset;             // Byte offset to grid payload
  uint64_t payloadSize;               // Size of grid payload in bytes
  int32_t  bboxMin[3];                // Bounding box min
  int32_t  bboxMax[3];                // Bounding box max
  uint32_t dims[3];                   // Grid dimensions
  uint32_t reserved2;
  char     name[56];                  // Grid name for debugging
};

CCL_NAMESPACE_END
