/* SPDX-FileCopyrightText: 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0 */

//#include "kernel/util/nanovdb.h"
#if defined(WITH_NANOVDB)
//#  include "kernel/util/nanovdb.h"
#  include <nanovdb/NanoVDB.h>
#endif

/* clang-format off */
struct ONEAPIKernelContext : public KernelGlobalsGPU {
  public:
#  if defined(WITH_GPU_CPUIMAGE)
#    include "kernel/device/cpu/image.h"
#  else
#    include "kernel/device/gpu/image.h"
#  endif
  /* clang-format on */
