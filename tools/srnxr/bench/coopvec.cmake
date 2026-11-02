# SPDX-FileCopyrightText: 2011-2025 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

# srnxr_coopvec: OptiX 9 cooperative vector probe, validation and benchmark. The OptiX
# programs are compiled to PTX and loaded at run time from next to the executable.

set(SRNXR_COOPVEC_PTX "${CMAKE_CURRENT_BINARY_DIR}/srnxr_coopvec_device.ptx")
set(SRNXR_COOPVEC_PTX_ARCH "compute_80" CACHE STRING "PTX target of the OptiX programs")
add_custom_command(
  OUTPUT ${SRNXR_COOPVEC_PTX}
  COMMAND ${CMAKE_CUDA_COMPILER} --ptx -arch=${SRNXR_COOPVEC_PTX_ARCH} -std=c++17
          -lineinfo -DSRNXR_STANDALONE=1
          -I "${SRNXR_OPTIX_ROOT}/include" -I "${CYCLES_SRC}" -I "${CMAKE_CURRENT_SOURCE_DIR}"
          -o ${SRNXR_COOPVEC_PTX} "${CMAKE_CURRENT_SOURCE_DIR}/srnxr_coopvec_device.cu"
  DEPENDS srnxr_coopvec_device.cu srnxr_coopvec_params.h srnxr_compat.h
          ${CYCLES_SRC}/kernel/util/image_3d_srnxr.h ${CYCLES_SRC}/kernel/util/image_3d_srnxr_decode.h
  COMMENT "OptiX programs of srnxr_coopvec")
add_custom_target(srnxr_coopvec_ptx ALL DEPENDS ${SRNXR_COOPVEC_PTX})

# Variant with the FP32 output products, to check whether the installed OptiX accepts them.
set(SRNXR_COOPVEC_PTX_FP32 "${CMAKE_CURRENT_BINARY_DIR}/srnxr_coopvec_device_fp32.ptx")
add_custom_command(
  OUTPUT ${SRNXR_COOPVEC_PTX_FP32}
  COMMAND ${CMAKE_CUDA_COMPILER} --ptx -arch=${SRNXR_COOPVEC_PTX_ARCH} -std=c++17
          -lineinfo -DSRNXR_STANDALONE=1 -DSRNXR_COOPVEC_TRY_FP32_OUTPUT=1
          -I "${SRNXR_OPTIX_ROOT}/include" -I "${CYCLES_SRC}" -I "${CMAKE_CURRENT_SOURCE_DIR}"
          -o ${SRNXR_COOPVEC_PTX_FP32} "${CMAKE_CURRENT_SOURCE_DIR}/srnxr_coopvec_device.cu"
  DEPENDS srnxr_coopvec_device.cu srnxr_coopvec_params.h srnxr_compat.h
          ${CYCLES_SRC}/kernel/util/image_3d_srnxr.h ${CYCLES_SRC}/kernel/util/image_3d_srnxr_decode.h
  COMMENT "OptiX programs of srnxr_coopvec, FP32 output variant")
add_custom_target(srnxr_coopvec_ptx_fp32 ALL DEPENDS ${SRNXR_COOPVEC_PTX_FP32})

add_executable(srnxr_coopvec srnxr_coopvec.cu ${CYCLES_SRC}/scene/srnxr_model.cpp)
add_dependencies(srnxr_coopvec srnxr_coopvec_ptx)
target_include_directories(srnxr_coopvec PRIVATE ${CYCLES_SRC} ${CMAKE_CURRENT_SOURCE_DIR}
                           "${SRNXR_OPTIX_ROOT}/include")
target_compile_definitions(srnxr_coopvec PRIVATE SRNXR_STANDALONE=1)
target_link_libraries(srnxr_coopvec PRIVATE ${CMAKE_DL_LIBS})
add_custom_command(TARGET srnxr_coopvec POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${SRNXR_COOPVEC_PTX} $<TARGET_FILE:srnxr_coopvec>_device.ptx)
