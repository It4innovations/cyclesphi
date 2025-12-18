cmake_minimum_required(VERSION 3.15)
project(nanovdb_multires_convert)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform-specific compiler flags
if(MSVC)
    # Windows - Visual Studio
    add_compile_options(/W3)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    # Linux and macOS - GCC/Clang
    add_compile_options(-Wall -Wextra -pedantic)
endif()

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

# Debug output
message(STATUS "TBB_LIBRARY_DIR: ${TBB_LIBRARY_DIR}")
message(STATUS "TBB_LIBRARY: ${TBB_LIBRARY}")
message(STATUS "OPENVDB_LIBRARY: ${OPENVDB_LIBRARY}")

# Add library directories to link path
link_directories(${TBB_LIBRARY_DIR})
link_directories(${OPENVDB_LIBRARY_DIR})

# NanoVDB is typically header-only and included with OpenVDB
set(NANOVDB_INCLUDE_DIR "" CACHE PATH "Path to NanoVDB include directory (if not with OpenVDB)")

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
