/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Launch parameters of the srnxr_coopvec OptiX pipeline. */
struct SRNXRCoopVecParams {
  const uint8_t *blob;
  const float4 *points;
  float *out;
  int n;
  int output_dim;
};
