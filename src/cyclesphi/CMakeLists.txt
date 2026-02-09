#####################################################################################################################
# Copyright(C) 2011-2025 IT4Innovations National Supercomputing Center, VSB - Technical University of Ostrava
#
# This program is free software : you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#####################################################################################################################

if(WITH_CLIENT_GPUJPEG)
  find_package(GPUJPEG REQUIRED)
endif()

if(WITH_CLIENT_MPI_SOCKET)
	find_package(MPI REQUIRED)
endif()

find_package(braas_hpc_renderengine REQUIRED)

#message("LIB: ${braas_hpc_renderengine_LIBRARIES}")
#message("INCLUDE: ${braas_hpc_renderengine_INCLUDE_DIR}")

set(INC
  ..
  #../../lib/braas-hpc-renderengine/src
  #../../third_party
  ../app
)

set(INC_SYS
   #${GPUJPEG_INCLUDE_DIR}
   #${braas_hpc_renderengine_INCLUDE_DIR}
   ${MPI_CXX_HEADER_DIR}
)

set(LIB
  cycles_device
  cycles_kernel
  cycles_scene
  cycles_session
  cycles_bvh
  cycles_subd
  cycles_graph
  cycles_util
  braas_hpc_renderengine::braas_hpc_renderengine
  #${braas_hpc_renderengine_LIBRARIES}
)

#find_package(CUDA REQUIRED)

#if(WITH_CLIENT_GPUJPEG)
#    list(APPEND LIB
#      ${GPUJPEG_LIBRARIES}
#    )
#endif()

# if(WITH_CYCLES_DEVICE_OPTIX OR WITH_CYCLES_DEVICE_CUDA)
#   if(WITH_CUDA_DYNLOAD)
#     list(APPEND LIB
#       extern_cuew
#     )
#   else()
#     list(APPEND LIB
#       ${CUDA_CUDA_LIBRARY}
#     )
#   endif()
# endif()

if(WITH_ALEMBIC)
  add_definitions(-DWITH_ALEMBIC)
  list(APPEND INC_SYS
    ${ALEMBIC_INCLUDE_DIRS}
  )
  list(APPEND LIB
    ${ALEMBIC_LIBRARIES}
  )
endif()

if(WITH_CYCLES_OSL)
  list(APPEND LIB cycles_kernel_osl)
endif()

if(CYCLES_STANDALONE_REPOSITORY)
  list(APPEND LIB extern_sky)
else()
  list(APPEND LIB bf_intern_sky)
endif()

if(WITH_CYCLES_STANDALONE AND WITH_CYCLES_STANDALONE_GUI)
  list(APPEND INC_SYS
    ${Epoxy_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
  )
  list(APPEND LIB ${Epoxy_LIBRARIES} ${SDL2_LIBRARIES})
endif()

if(WITH_USD)
  # Silence warning from USD headers using deprecated TBB header.
  add_definitions(
    -D__TBB_show_deprecation_message_atomic_H
    -D__TBB_show_deprecation_message_task_H
  )

  list(APPEND INC_SYS
    ${USD_INCLUDE_DIRS}
  )
  list(APPEND LIB
    cycles_hydra
    ${USD_LIBRARIES}
  )
endif()

cycles_external_libraries_append(LIB)

# Common configuration.

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

if(WITH_CLIENT_GPUJPEG)
	add_definitions(-DWITH_CLIENT_GPUJPEG)
endif()

if(WITH_CLIENT_MPI_SOCKET)
	add_definitions(-DWITH_CLIENT_MPI_SOCKET)
endif()

# if(WITH_CYCLES_DEVICE_ONEAPI)
#   include_directories(../third_party)
# endif()

# Add OpenMP support
find_package(OpenMP REQUIRED)

# Application build targets
###############cyclesphi############################
if(WITH_CYCLES_STANDALONE)

  # Application build targets
  set(SRC
      cyclesphi.cpp
      
      cyclesphi_common.cpp
      cyclesphi_common.h
      
      #frame_output_driver.cpp
      #frame_output_driver.h

      frame_display_driver.cpp
      frame_display_driver.h
      
      #../../lib/braas-hpc-renderengine/src/renderengine_tcp.cpp
      #../../lib/braas-hpc-renderengine/src/renderengine_tcp.h

      ../app/cycles_xml_bin.cpp
      ../app/cycles_xml_bin.h
  )

  add_executable(cyclesphi ${SRC} ${INC} ${INC_SYS})
  unset(SRC)

  target_link_libraries(cyclesphi PRIVATE ${LIB})

  if(OpenMP_FOUND)
    target_link_libraries(cyclesphi PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(cyclesphi PRIVATE ${OpenMP_CXX_FLAGS})
  endif()

  if(CYCLES_STANDALONE_REPOSITORY)
  cycles_install_libraries("")
  endif()

  install(
  TARGETS cyclesphi
  DESTINATION ${CMAKE_INSTALL_PREFIX})

endif()
###############cyclesphi_space############################
if(WITH_CYCLES_STANDALONE)

  # Application build targets
  set(SRC
      cyclesphi_space.cpp
      
      cyclesphi_common.cpp
      cyclesphi_common.h    
      
      #frame_output_driver.cpp
      #frame_output_driver.h
      
      frame_display_driver.cpp
      frame_display_driver.h

      #../../lib/braas-hpc-renderengine/src/renderengine_tcp.cpp
      #../../lib/braas-hpc-renderengine/src/renderengine_tcp.h

      ../app/cycles_xml_bin.cpp
      ../app/cycles_xml_bin.h
  )

  add_executable(cyclesphi_space ${SRC} ${INC} ${INC_SYS})
  unset(SRC)

  target_link_libraries(cyclesphi_space PRIVATE ${LIB})

  if(OpenMP_FOUND)
    target_link_libraries(cyclesphi_space PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(cyclesphi_space PRIVATE ${OpenMP_CXX_FLAGS})
  endif()

  install(
  TARGETS cyclesphi_space
  DESTINATION ${CMAKE_INSTALL_PREFIX})

endif()
###############cyclesphi_mpi############################
if(WITH_CYCLES_STANDALONE)
  if(WITH_CLIENT_MPI_SOCKET)
    set(SRC
        cyclesphi_mpi.cpp
        
        cyclesphi_common.cpp
        cyclesphi_common.h    
        
        #frame_output_driver.cpp
        #frame_output_driver.h
        
        frame_display_driver.cpp
        frame_display_driver.h

        #../../lib/braas-hpc-renderengine/src/renderengine_tcp.cpp
        #../../lib/braas-hpc-renderengine/src/renderengine_tcp.h

        ../app/cycles_xml_bin.cpp
        ../app/cycles_xml_bin.h
    )

    add_executable(cyclesphi_mpi ${SRC} ${INC} ${INC_SYS})
    unset(SRC)

    target_link_libraries(cyclesphi_mpi PRIVATE ${LIB} ${MPI_msmpi_LIBRARY} ${MPI_LIBRARIES})

    if(OpenMP_FOUND)
      target_link_libraries(cyclesphi_mpi PRIVATE OpenMP::OpenMP_CXX)
      target_compile_options(cyclesphi_mpi PRIVATE ${OpenMP_CXX_FLAGS})
    endif()

    install(
    TARGETS cyclesphi_mpi
    DESTINATION ${CMAKE_INSTALL_PREFIX})
  endif()
endif()