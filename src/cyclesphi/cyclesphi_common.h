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
		used_device("CPU"), 		
		use_mpi(false),    
		world_rank(0),
		world_size(1),
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
};

void session_init(FromCL& fromCL, Options &options, int session_id);
void session_exit(FromCL& fromCL, Options& options);

int cyclesphi(int ac, char** av, TcpConnection* blenderClientTcp, FromCL& fromCL, std::vector<Options>& options);
