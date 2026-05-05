cmake_minimum_required(VERSION 3.15)
project(cyclesphi_tools_nanovdb)

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

# Find ZFP package using config file
if(WITH_ZFP)
    find_package(zfp REQUIRED CONFIG)
    message(STATUS "Found ZFP: ${zfp_DIR}")
endif()

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
##################convert_vdb_to_nanovdb_multires executable configuration##################
# Create executable
add_executable(convert_vdb_to_nanovdb_multires convert_vdb_to_nanovdb_multires.cpp)

# Include directories
target_include_directories(convert_vdb_to_nanovdb_multires PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(convert_vdb_to_nanovdb_multires PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(convert_vdb_to_nanovdb_multires PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(convert_vdb_to_nanovdb_multires PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(convert_vdb_to_nanovdb_multires PRIVATE pthread)
endif()

# Set output directory
set_target_properties(convert_vdb_to_nanovdb_multires PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS convert_vdb_to_nanovdb_multires
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building convert_vdb_to_nanovdb_multires")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################convert_vdb_to_nanovdb_derivates executable configuration##################
# Create executable
add_executable(convert_vdb_to_nanovdb_derivates convert_vdb_to_nanovdb_derivates.cpp)

# Include directories
target_include_directories(convert_vdb_to_nanovdb_derivates PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Define Cycles namespace macros needed by kernel headers
target_compile_definitions(convert_vdb_to_nanovdb_derivates PRIVATE
    CCL_NAMESPACE_BEGIN=
    CCL_NAMESPACE_END=
)

# Remove pedantic flag and allow permissive compilation for this target
# to avoid conflicts with Cycles math functions and stdlib
target_compile_options(convert_vdb_to_nanovdb_derivates PRIVATE
    -Wno-pedantic
    -fpermissive
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(convert_vdb_to_nanovdb_derivates PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(convert_vdb_to_nanovdb_derivates PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(convert_vdb_to_nanovdb_derivates PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(convert_vdb_to_nanovdb_derivates PRIVATE pthread)
endif()

# Set output directory
set_target_properties(convert_vdb_to_nanovdb_derivates PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS convert_vdb_to_nanovdb_derivates
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building convert_vdb_to_nanovdb_derivates")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")

#####################################################################################
##################convert_vdb_to_nanovdb_derivates_vec4 executable configuration##################
# Create executable
add_executable(convert_vdb_to_nanovdb_derivates_vec4 convert_vdb_to_nanovdb_derivates_vec4.cpp)

# Include directories
target_include_directories(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Define Cycles namespace macros needed by kernel headers
target_compile_definitions(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE
    CCL_NAMESPACE_BEGIN=
    CCL_NAMESPACE_END=
)

# Remove pedantic flag and allow permissive compilation for this target
# to avoid conflicts with Cycles math functions and stdlib
target_compile_options(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE
    -Wno-pedantic
    -fpermissive
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(convert_vdb_to_nanovdb_derivates_vec4 PRIVATE pthread)
endif()

# Set output directory
set_target_properties(convert_vdb_to_nanovdb_derivates_vec4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS convert_vdb_to_nanovdb_derivates_vec4
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building convert_vdb_to_nanovdb_derivates_vec4")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################downsample_vdb executable configuration##################
# Create executable
add_executable(downsample_vdb downsample_vdb.cpp)

# Include directories
target_include_directories(downsample_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(downsample_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Enable OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(downsample_vdb PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Link libraries
target_link_libraries(downsample_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(downsample_vdb PRIVATE OpenMP::OpenMP_CXX)
endif()

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(downsample_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(downsample_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(downsample_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS downsample_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building downsample_vdb")
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
##################compression_zfp_to_vdb executable configuration##################
if(WITH_ZFP)
    # Create executable
    add_executable(compression_zfp_to_vdb compression_zfp_to_vdb.cpp)

    # Include directories
    target_include_directories(compression_zfp_to_vdb PRIVATE
        ${OPENVDB_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIRS}
    )

    # Add NanoVDB include directory if specified separately
    if(NANOVDB_INCLUDE_DIR)
        target_include_directories(compression_zfp_to_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
    endif()

    # Link libraries
    target_link_libraries(compression_zfp_to_vdb PRIVATE
        ${OPENVDB_LIBRARY}
        ${TBB_LIBRARY}
        zfp::zfp
    )

    # Platform-specific linking
    if(UNIX AND NOT APPLE)
        # Linux specific
        target_link_libraries(compression_zfp_to_vdb PRIVATE pthread dl)
    elseif(APPLE)
        # macOS specific
        target_link_libraries(compression_zfp_to_vdb PRIVATE pthread)
    endif()

    # Set output directory
    set_target_properties(compression_zfp_to_vdb PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Installation
    install(TARGETS compression_zfp_to_vdb
        RUNTIME DESTINATION bin
    )

    # Print configuration info
    message(STATUS "Building compression_zfp_to_vdb")
    message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()


#####################################################################################
##################convert_derivates_bin_to_dense_vdb executable configuration##################
# Create executable
add_executable(convert_derivates_bin_to_dense_vdb convert_derivates_bin_to_dense_vdb.cpp)

# Include directories
target_include_directories(convert_derivates_bin_to_dense_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# Define Cycles namespace macros needed by kernel headers
target_compile_definitions(convert_derivates_bin_to_dense_vdb PRIVATE
    CCL_NAMESPACE_BEGIN=
    CCL_NAMESPACE_END=
    USE_OPENMP
)

# Remove pedantic flag and allow permissive compilation for this target
# to avoid conflicts with Cycles math functions and stdlib
target_compile_options(convert_derivates_bin_to_dense_vdb PRIVATE
    -Wno-pedantic
    -fpermissive
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(convert_derivates_bin_to_dense_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Enable OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(convert_derivates_bin_to_dense_vdb PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Link libraries
target_link_libraries(convert_derivates_bin_to_dense_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
    OpenMP::OpenMP_CXX
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(convert_derivates_bin_to_dense_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(convert_derivates_bin_to_dense_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(convert_derivates_bin_to_dense_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS convert_derivates_bin_to_dense_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building convert_derivates_bin_to_dense_vdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################compression_hierarchical_to_vdb executable configuration##################
# Create executable
add_executable(compression_hierarchical_to_vdb compression_hierarchical_to_vdb.cpp)

# Include directories
target_include_directories(compression_hierarchical_to_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(compression_hierarchical_to_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(compression_hierarchical_to_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(compression_hierarchical_to_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(compression_hierarchical_to_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(compression_hierarchical_to_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS compression_hierarchical_to_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building compression_hierarchical_to_vdb")
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


#####################################################################################
##################raw_to_openvdb executable configuration##################
# Create executable
add_executable(raw_to_openvdb raw_to_openvdb.cpp)

# Include directories
target_include_directories(raw_to_openvdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(raw_to_openvdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(raw_to_openvdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(raw_to_openvdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(raw_to_openvdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(raw_to_openvdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS raw_to_openvdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building raw_to_openvdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")


#####################################################################################
##################merge_vdb executable configuration##################
# Create executable
add_executable(merge_vdb merge_vdb.cpp)

# Include directories
target_include_directories(merge_vdb PRIVATE
    ${OPENVDB_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# Add NanoVDB include directory if specified separately
if(NANOVDB_INCLUDE_DIR)
    target_include_directories(merge_vdb PRIVATE ${NANOVDB_INCLUDE_DIR})
endif()

# Enable OpenMP
if(OpenMP_CXX_FOUND)
    target_compile_options(merge_vdb PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Link libraries
target_link_libraries(merge_vdb PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(merge_vdb PRIVATE OpenMP::OpenMP_CXX)
endif()

# Platform-specific linking
if(UNIX AND NOT APPLE)
    # Linux specific
    target_link_libraries(merge_vdb PRIVATE pthread dl)
elseif(APPLE)
    # macOS specific
    target_link_libraries(merge_vdb PRIVATE pthread)
endif()

# Set output directory
set_target_properties(merge_vdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS merge_vdb
    RUNTIME DESTINATION bin
)

# Print configuration info
message(STATUS "Building merge_vdb")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
