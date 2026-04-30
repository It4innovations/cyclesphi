cmake_minimum_required(VERSION 3.15)
project(nanovdb_multires_convert)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform-specific compiler flags
if(MSVC)
    # Windows - Visual Studio
    add_compile_options(/W3 /bigobj)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    # Linux and macOS - GCC/Clang
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Options
option(WITH_ZFP "Build ZFP compression tools" OFF)

# Blender library path configuration
set(BLENDER_LIB_DIR "" CACHE PATH "Path to Blender lib directory (e.g., blender/lib/windows_x64)")
message(STATUS "Using Blender libraries from: ${BLENDER_LIB_DIR}")

# Set up OpenVDB paths
set(OPENVDB_ROOT "${BLENDER_LIB_DIR}/openvdb")
set(TBB_ROOT "${BLENDER_LIB_DIR}/tbb")
set(ZLIB_ROOT "${BLENDER_LIB_DIR}/zlib")

# Include directories
set(OPENVDB_INCLUDE_DIRS "${OPENVDB_ROOT}/include")
set(TBB_INCLUDE_DIRS "${TBB_ROOT}/include")

# Library directories
if(WIN32)
    set(OPENVDB_LIBRARY_DIR "${OPENVDB_ROOT}/lib")
    set(TBB_LIBRARY_DIR "${TBB_ROOT}/lib")
    set(ZLIB_LIBRARY_DIR "${ZLIB_ROOT}/lib")
else()
    set(OPENVDB_LIBRARY_DIR "${OPENVDB_ROOT}/lib")
    set(TBB_LIBRARY_DIR "${TBB_ROOT}/lib")
    set(ZLIB_LIBRARY_DIR "${ZLIB_ROOT}/lib")
endif()

# Find libraries
find_library(OPENVDB_LIBRARY NAMES openvdb libopenvdb PATHS ${OPENVDB_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(TBB_LIBRARY NAMES tbb libtbb PATHS ${TBB_LIBRARY_DIR} NO_DEFAULT_PATH)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Debug output
message(STATUS "TBB_LIBRARY_DIR: ${TBB_LIBRARY_DIR}")
message(STATUS "TBB_LIBRARY: ${TBB_LIBRARY}")
message(STATUS "OPENVDB_LIBRARY: ${OPENVDB_LIBRARY}")

# Add library directories to link path
link_directories(${TBB_LIBRARY_DIR})
link_directories(${OPENVDB_LIBRARY_DIR})

# NanoVDB is typically header-only and included with OpenVDB
set(NANOVDB_INCLUDE_DIR "" CACHE PATH "Path to NanoVDB include directory (if not with OpenVDB)")

#####################################################################################
##################nanovdb_multires_convert executable configuration##################
# Create executable
add_executable(nanovdb_multires_convert nanovdb_multires_convert.cpp)

# Include directories
target_include_directories(nanovdb_multires_convert PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(nanovdb_multires_convert PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(nanovdb_multires_convert PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(nanovdb_multires_convert PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(nanovdb_multires_convert PRIVATE pthread)
endif()

# Set output directory
set_target_properties(nanovdb_multires_convert PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS nanovdb_multires_convert
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building nanovdb_multires_convert")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################nanovdb_derivates_convert executable configuration##################
# Create executable
add_executable(nanovdb_derivates_convert nanovdb_derivates_convert.cpp)

# Include directories
target_include_directories(nanovdb_derivates_convert PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Define Cycles namespace macros needed by kernel headers
target_compile_definitions(nanovdb_derivates_convert PRIVATE
    CCL_NAMESPACE_BEGIN=
    CCL_NAMESPACE_END=
)

# Remove pedantic flag and allow permissive compilation for this target
# to avoid conflicts with Cycles math functions and stdlib
target_compile_options(nanovdb_derivates_convert PRIVATE
    -Wno-pedantic
    -fpermissive
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(nanovdb_derivates_convert PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(nanovdb_derivates_convert PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(nanovdb_derivates_convert PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(nanovdb_derivates_convert PRIVATE pthread)
endif()

# Set output directory
set_target_properties(nanovdb_derivates_convert PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS nanovdb_derivates_convert
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building nanovdb_derivates_convert")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")

#####################################################################################
##################nanovdb_derivates_convert_vec4 executable configuration##################
# Create executable
add_executable(nanovdb_derivates_convert_vec4 nanovdb_derivates_convert_vec4.cpp)

# Include directories
target_include_directories(nanovdb_derivates_convert_vec4 PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Define Cycles namespace macros needed by kernel headers
target_compile_definitions(nanovdb_derivates_convert_vec4 PRIVATE
    CCL_NAMESPACE_BEGIN=
    CCL_NAMESPACE_END=
)

# Remove pedantic flag and allow permissive compilation for this target
# to avoid conflicts with Cycles math functions and stdlib
target_compile_options(nanovdb_derivates_convert_vec4 PRIVATE
    -Wno-pedantic
    -fpermissive
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(nanovdb_derivates_convert_vec4 PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(nanovdb_derivates_convert_vec4 PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(nanovdb_derivates_convert_vec4 PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(nanovdb_derivates_convert_vec4 PRIVATE pthread)
endif()

# Set output directory
set_target_properties(nanovdb_derivates_convert_vec4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS nanovdb_derivates_convert_vec4
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building nanovdb_derivates_convert_vec4")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################downsampled_vdb executable configuration##################
# Create executable
add_executable(downsampled_vdb downsampled_vdb.cpp)

# Include directories
target_include_directories(downsampled_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(downsampled_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(downsampled_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(downsampled_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(downsampled_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(downsampled_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS downsampled_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building downsampled_vdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################compare_vdb executable configuration##################
# Create executable
add_executable(compare_vdb compare_vdb.cpp)

# Include directories
target_include_directories(compare_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(compare_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(compare_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(compare_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(compare_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(compare_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS compare_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building compare_vdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################compressed_zfp_to_vdb executable configuration##################
if(WITH_ZFP)
    # Create executable
    add_executable(compressed_zfp_to_vdb compressed_zfp_to_vdb.cpp)

    # Include directories
    target_include_directories(compressed_zfp_to_vdb PRIVATE
        ${OPENVDB_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIRS}
    )

    # Add NanoVDB include directory if specified separately
    if(NANOVDB_INCLUDE_DIR)
        target_include_directories(compressed_zfp_to_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
    endif()

    # Link libraries
    target_link_libraries(compressed_zfp_to_vdb PRIVATE
        ${OPENVDB_LIBRARY}
        ${TBB_LIBRARY}
    )

    # Platform-specific linking
    if(UNIX AND NOT APPLE)
        # Linux specific
        target_link_libraries(compressed_zfp_to_vdb PRIVATE pthread dl)
    elseif(APPLE)
        # macOS specific
        target_link_libraries(compressed_zfp_to_vdb PRIVATE pthread)
    endif()

    # Set output directory
    set_target_properties(compressed_zfp_to_vdb PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Installation
    install(TARGETS compressed_zfp_to_vdb
        RUNTIME DESTINATION bin
    )

    # Print configuration info
    message(STATUS "Building compressed_zfp_to_vdb")
    message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()


#####################################################################################
##################derivates_bin_to_dense_vdb executable configuration##################
# Create executable
add_executable(derivates_bin_to_dense_vdb derivates_bin_to_dense_vdb.cpp)

# Include directories
target_include_directories(derivates_bin_to_dense_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Define Cycles namespace macros needed by kernel headers
target_compile_definitions(derivates_bin_to_dense_vdb PRIVATE
    CCL_NAMESPACE_BEGIN=
    CCL_NAMESPACE_END=
    USE_OPENMP
)

# Remove pedantic flag and allow permissive compilation for this target
# to avoid conflicts with Cycles math functions and stdlib
target_compile_options(derivates_bin_to_dense_vdb PRIVATE
    -Wno-pedantic
    -fpermissive
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(derivates_bin_to_dense_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Enable OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(derivates_bin_to_dense_vdb PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Link libraries
target_link_libraries(derivates_bin_to_dense_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
    OpenMP::OpenMP_CXX
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(derivates_bin_to_dense_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(derivates_bin_to_dense_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(derivates_bin_to_dense_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS derivates_bin_to_dense_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building derivates_bin_to_dense_vdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################hierarchical_compression_vdb executable configuration##################
# Create executable
add_executable(hierarchical_compression_vdb hierarchical_compression_vdb.cpp)

# Include directories
target_include_directories(hierarchical_compression_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(hierarchical_compression_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(hierarchical_compression_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(hierarchical_compression_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(hierarchical_compression_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(hierarchical_compression_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS hierarchical_compression_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building hierarchical_compression_vdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################openvdb_to_nanovdb executable configuration##################
# Create executable
add_executable(openvdb_to_nanovdb openvdb_to_nanovdb.cpp)

# Include directories
target_include_directories(openvdb_to_nanovdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(openvdb_to_nanovdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(openvdb_to_nanovdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(openvdb_to_nanovdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(openvdb_to_nanovdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(openvdb_to_nanovdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS openvdb_to_nanovdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building openvdb_to_nanovdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")

#####################################################################################
##################vdb_print executable configuration##################
# Create executable
add_executable(vdb_print vdb_print.cpp)

# Include directories
target_include_directories(vdb_print PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(vdb_print PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(vdb_print PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(vdb_print PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(vdb_print PRIVATE pthread)
endif()

# Set output directory
set_target_properties(vdb_print PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS vdb_print
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building vdb_print")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")

#####################################################################################
##################nanovdb_print executable configuration##################
# Create executable
add_executable(nanovdb_print nanovdb_print.cpp)

# Include directories
target_include_directories(nanovdb_print PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(nanovdb_print PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(nanovdb_print PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(nanovdb_print PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(nanovdb_print PRIVATE pthread)
endif()

# Set output directory
set_target_properties(nanovdb_print PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS nanovdb_print
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building nanovdb_print")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
