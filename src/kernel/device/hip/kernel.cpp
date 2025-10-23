/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* HIP kernel entry points */

#ifdef __HIP_DEVICE_COMPILE__

#  include "kernel/device/hip/compat.h"
#  include "kernel/device/hip/config.h"
#  include "kernel/device/hip/globals.h"

#  if defined(WITH_GPU_CPUIMAGE)
#    include "kernel/device/cpu/image.h"
#  else
#    include "kernel/device/gpu/image.h"
#  endif

#  include "kernel/device/gpu/kernel.h"

#endif
