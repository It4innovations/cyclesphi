// #####################################################################################################################
// # Copyright(C) 2011-2025 IT4Innovations National Supercomputing Center, VSB - Technical University of Ostrava
// #
// # This program is free software : you can redistribute it and/or modify
// # it under the terms of the GNU General Public License as published by
// # the Free Software Foundation, either version 3 of the License, or
// # (at your option) any later version.
// #
// # This program is distributed in the hope that it will be useful,
// # but WITHOUT ANY WARRANTY; without even the implied warranty of
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
// # GNU General Public License for more details.
// #
// # You should have received a copy of the GNU General Public License
// # along with this program.  If not, see <https://www.gnu.org/licenses/>.
// #
// #####################################################################################################################

#pragma once

// Prevent CUDA from declaring 'half' in global namespace to avoid conflict with Imath's half
#define CUDA_NO_HALF

// Prevent CUDA runtime headers from being included to avoid conflict with CUEW
// CUEW provides its own CUDA definitions for dynamic loading
#define __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_API_H__
#define __DRIVER_TYPES_H__
#define __VECTOR_TYPES_H__

// Define CUDA macros for non-CUDA compilation contexts
// These are needed when CUDA headers are indirectly included via third-party libraries
#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __inline__ inline
#define __CUDA_HOSTDEVICE__
#define __CUDA_HOSTDEVICE_FP16_DECL__ static inline
#define __VECTOR_FUNCTIONS_DECL__ static inline

// Define dim3 struct for non-CUDA compilation
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Define CUDA vector types for non-CUDA compilation
struct char1 { signed char x; };
struct uchar1 { unsigned char x; };
struct char2 { signed char x, y; };
struct uchar2 { unsigned char x, y; };
struct char3 { signed char x, y, z; };
struct uchar3 { unsigned char x, y, z; };
struct char4 { signed char x, y, z, w; };
struct uchar4 { unsigned char x, y, z, w; };

struct short1 { short x; };
struct ushort1 { unsigned short x; };
struct short2 { short x, y; };
struct ushort2 { unsigned short x, y; };
struct short3 { short x, y, z; };
struct ushort3 { unsigned short x, y, z; };
struct short4 { short x, y, z, w; };
struct ushort4 { unsigned short x, y, z, w; };

struct int1 { int x; };
struct uint1 { unsigned int x; };
struct int2 { int x, y; };
struct uint2 { unsigned int x, y; };
struct int3 { int x, y, z; };
struct uint3 { unsigned int x, y, z; };
struct int4 { int x, y, z, w; };
struct uint4 { unsigned int x, y, z, w; };

struct long1 { long int x; };
struct ulong1 { unsigned long int x; };
struct long2 { long int x, y; };
struct ulong2 { unsigned long int x, y; };
struct long3 { long int x, y, z; };
struct ulong3 { unsigned long int x, y, z; };
struct long4 { long int x, y, z, w; };
struct ulong4 { unsigned long int x, y, z, w; };

struct float1 { float x; };
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct longlong1 { long long int x; };
struct ulonglong1 { unsigned long long int x; };
struct longlong2 { long long int x, y; };
struct ulonglong2 { unsigned long long int x, y; };
struct longlong3 { long long int x, y, z; };
struct ulonglong3 { unsigned long long int x, y, z; };
struct longlong4 { long long int x, y, z, w; };
struct ulonglong4 { unsigned long long int x, y, z, w; };

struct double1 { double x; };
struct double2 { double x, y; };
struct double3 { double x, y, z; };
struct double4 { double x, y, z, w; };

// Define CUDA stream type for non-CUDA compilation
typedef struct CUstream_st* cudaStream_t;
typedef cudaStream_t gpuStream_t;
#endif

#include <stdio.h>
#include <atomic>

#include "session/buffers.h"
#include "session/session.h"

//#include "frame_output_driver.h"
#include "frame_display_driver.h"

#include "renderengine_tcp.h"

class FromCL {
public:
	FromCL(): 
		port(7000), 
		anim(-1), 
		use_anim(false), 
		filepath(),
		used_device("CPU"), 		
		use_mpi(false),    
		world_rank(0),
		world_size(1),
		threads(0),
#ifdef WITH_CLIENT_GPUJPEG
		use_gpujpeg(true),
#else
		use_gpujpeg(false),
#endif
    render_running(true)
	{
	}

	int port;
	int anim;
	bool use_anim;
	std::string filepath;
	std::string used_device;

	bool use_mpi;
	int world_rank;
	int world_size;

	int threads;

#ifdef WITH_CLIENT_GPUJPEG
	bool use_gpujpeg = true;
#else
	bool use_gpujpeg = false;
#endif

	// Atomic flag to control the infinite loops
	std::atomic<bool> render_running;

	virtual void parse_args(int argc, char** argv);
	virtual void usage();
};

struct Options {
	int id = 0;

	ccl::Session* session = nullptr;
	ccl::Scene* scene = nullptr;
	std::string filepath;
	int width, height;
	ccl::SceneParams scene_params;
	ccl::SessionParams session_params;
	bool quiet;
	bool show_help, interactive, pause;
	//std::string output_filepath;
	std::string output_pass;
	int session_samples = 0;

	//ccl::FrameOutputDriver* output_driver = nullptr;
	ccl::FrameDisplayDriver* display_driver = nullptr;

#ifdef WITH_CLIENT_GPUJPEG
	bool use_gpujpeg = true;
#else
	bool use_gpujpeg = false;
#endif
};

void session_init(FromCL& fromCL, Options &options, int session_id);
void session_exit(FromCL& fromCL, Options& options);

int cyclesphi(int ac, char** av, TcpConnection* blenderClientTcp, FromCL& fromCL, std::vector<Options>& options);
