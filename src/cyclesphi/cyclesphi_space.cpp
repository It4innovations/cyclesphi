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

#include "cyclesphi_common.h"
#include "cycles_xml_bin.h"


class FromCLSpace: public FromCL {
public:
	FromCLSpace() : FromCL(), space_port(6000), space_server("localhost"), space_server_port(5005) {
	}
	int space_port;
	std::string space_server;
	int space_server_port;

	void usage() override
	{
		std::cout << "./cyclesphi_space <options>" << std::endl;
	
		std::cout << "options:" << std::endl;
		std::cout << "\t--scene X" << std::endl;
		std::cout << "\t--device X" << std::endl;
		std::cout << "\t--port X" << std::endl;
		std::cout << "\t--anim X" << std::endl;

		std::cout << "\t--space-port X" << std::endl;
		std::cout << "\t--space-server X" << std::endl;
		std::cout << "\t--space-server-port X" << std::endl;
	
		exit(0);
	}
	
	void parse_args(int argc, char** argv) override
	{
		//if (argc < 2) {
		//	usage();
		//}
	
		for (int i = 1; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--port") {
				port = std::stoi(argv[++i]);
			}
			else if (arg == "--anim") {
				use_anim = true;
				anim = std::stoi(argv[++i]);
			}
			else if (arg == "--scene") {
				filepath = argv[++i];
			}
			else if (arg == "--device") {
				used_device = argv[++i];
			}
			else if (arg == "--space-port") {
				space_port = std::stoi(argv[++i]);
			}
			else if (arg == "--space-server") {
				space_server = argv[++i];
			}
			else if (arg == "--space-server-port") {
				space_server_port = std::stoi(argv[++i]);
			}						
			else if (arg == "-h" || arg == "--help") {
				usage();
			}
		}
	}	
};

// Semaphore class
class Semaphore {
public:
	Semaphore(int count = 0)
		: count(count) {}

	void notify() {
		std::unique_lock<std::mutex> lock(mtx);
		++count;
		cv.notify_one();
	}

	void wait() {
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [this] { return count > 0; });
		--count;
	}

private:
	std::mutex mtx;
	std::condition_variable cv;
	int count;
};

// Semaphores for function1 and function2
Semaphore sem_render(0);
Semaphore sem_bspace(0);

////////////////////////
//float grid_transform = 1;
//int value_convert = 0;
//float object_size = 1000.0f;
//
//float min_value = 0;
//float max_value = 1;
//float min_value_reduced = 0;
//float max_value_reduced = 1;
//
//int use_particle_length = 0;
//float filter_min = FLT_MIN;
//float filter_max = FLT_MAX;
//
//size_t particles_count = 0;
//size_t voxels_count = 0;
//
//float bbox_min[3] = { 0,0,0 };
//float bbox_max[3] = { 100,100,100 };
//int bbox_dim = 100;
//
//int bbox_min_orig_local[3];
//int bbox_max_orig_local[3];
//
//int bbox_min_orig[3];
//int bbox_max_orig[3];
//
///////////////////////////	
//int particle_type = 0;
//int block_name_id = 0;

// file_type_items

// SpaceData class stores and manages configuration and metadata for spatial data conversions.
class SpaceData {
public:
	// Enumeration for message types used in communication or processing.
	enum MessageType {
		eExit = -1,  // Signal to exit the process.
		eInfo = 1,   // Informational message.
		eData = 2    // Data processing message.
	};

	// Enumeration for value conversion types.
	//enum ValueConvertType {
	//	eDefault = 0 // Default conversion type.
	//};

	// Enumeration for density calculation methods.
	//enum DenseType {
	//	eNone = 0,     // No density computation.
	//	eType1 = 1,
	//	eType2 = 2,
	//	eType3 = 3,
	//	eType4 = 4,
	//	eType5 = 5,
	//	eType6 = 6,
	//	eType7 = 7,
	//};


public:
	// Public member variables to store configuration and metadata.
	int message_type;        // Type of the message (see MessageType).
	int particle_type;       // Type of particle being processed.
	int block_name_id;       // Identifier for the block name.

	float grid_transform;    // Transformation applied to the grid.

	//int value_convert;       // Type of value conversion (see ValueConvertType).
	int extracted_type;      // Type of extracted particle (see ExtractedType).
	int dense_type;          // Type of density calculation (see DenseType).
	int dense_norm;          // Type of density calculation (see DenseNorm).

	float object_size;       // Size of the object in the grid.

	float min_value;         // Minimum value in the data.
	float max_value;         // Maximum value in the data.

	float min_rho;           // Minimum density value.
	float max_rho;           // Maximum density value.

	float min_value_reduced; // Reduced minimum value after processing.
	float max_value_reduced; // Reduced maximum value after processing.

	float particle_fix_size; // Flag indicating whether particle length is used.
	float filter_min;        // Minimum filter value.
	float filter_max;        // Maximum filter value.

	size_t particles_count;  // Total number of particles.
	size_t voxels_count;     // Total number of voxels.

	float bbox_min[3];       // Minimum bounding box coordinates (x, y, z).
	float bbox_max[3];       // Maximum bounding box coordinates (x, y, z).
	int bbox_dim;            // Dimensions of the bounding box.

	int bbox_min_orig_local[3]; // Local original minimum bounding box coordinates.
	int bbox_max_orig_local[3]; // Local original maximum bounding box coordinates.

	int bbox_min_orig[3];    // Original minimum bounding box coordinates.
	int bbox_max_orig[3];    // Original maximum bounding box coordinates.

	double box_size;         // Size of the bounding box.

	float min_value_local;   // Local minimum value.
	float max_value_local;   // Local maximum value.
	size_t particles_count_local; // Local particle count.

	double transform_scale;  // Scale transformation applied to the data.

	int frame;
	int anim_type;
	int anim_task_counter;
	std::string full_filepath; //path to saved vdb

public:
	SpaceData():SpaceData(0, 0, 200) {
	}
	SpaceData(int export_type, int export_dataset, int grid_dim) {
		message_type = 0;
		particle_type = export_type;
		block_name_id = export_dataset;

		grid_transform = 1.0f;
		//value_convert = 0;
		dense_type = 0;
		dense_norm = 0;
		object_size = 1000.0f;

		min_value = 0.0f;
		max_value = 1.0f;

		min_rho = 0.0f;
		max_rho = 1.0f;

		min_value_reduced = 0.0f;
		max_value_reduced = 1.0f;

		particle_fix_size = 0.0f;
		filter_min = -FLT_MAX;
		filter_max = FLT_MAX;

		particles_count = 0;
		voxels_count = 0;

		memset(bbox_min, 0, sizeof(bbox_min));

		bbox_max[0] = 1000;
		bbox_max[1] = 1000;
		bbox_max[2] = 1000;

		bbox_dim = grid_dim;

		memset(bbox_min_orig_local, 0, sizeof(bbox_min_orig_local));
		memset(bbox_max_orig_local, 0, sizeof(bbox_max_orig_local));

		memset(bbox_min_orig, 0, sizeof(bbox_min_orig));
		memset(bbox_max_orig, 0, sizeof(bbox_max_orig));

		box_size = 0.0;

		min_value_local = 0.0f;
		max_value_local = 0.0f;
		particles_count_local = 0;

		transform_scale = 0.0;
		frame = 0;
		anim_type = 0;
		anim_task_counter = 0;
	}
};
SpaceData spaceData;
int file_type = FTI_NONE;
////////////////////////
ccl::vector<char> grid_handle_final;


// Converts a void pointer to its string representation
std::string voidToStr(void* pointer) {
	std::ostringstream oss;
	oss << std::hex << pointer;
	return oss.str();
}

// Converts a size_t value to its string representation
std::string uint64ToStr(size_t value) {
	return std::to_string(value);
}

////////////////////////////////////////////BSPACE/////////////////////////////////////////////////
void bspace_loop(FromCLSpace &fromCL)
{
	while (true) {
		TcpConnection bSpaceClientTcp;
		bSpaceClientTcp.init_sockets_data("localhost", fromCL.space_port);

		TcpConnection spaceConverterServerTcp;
		spaceConverterServerTcp.init_sockets_data(fromCL.space_server.c_str(), fromCL.space_server_port, false);

		int message_type = 0;
		while (true) {
			bSpaceClientTcp.recv_data_data((char*)&message_type, sizeof(message_type), false);
			if (bSpaceClientTcp.is_error()) {
				message_type = -1;
			}

			spaceConverterServerTcp.send_data_data((char*)&message_type, sizeof(message_type), false);
			if (spaceConverterServerTcp.is_error()) {
				message_type = -1;
			}

			printf("message_type: %d\n", message_type); fflush(0);

			if (message_type == -1) {
				break;
			}

			if (message_type == 1)
			{
				int anim_type = 0;
				spaceConverterServerTcp.recv_data_data((char*)&anim_type, sizeof(int), false);
				bSpaceClientTcp.send_data_data((char*)&anim_type, sizeof(int), false);

				int anim_start = 0;
				spaceConverterServerTcp.recv_data_data((char*)&anim_start, sizeof(int), false);
				bSpaceClientTcp.send_data_data((char*)&anim_start, sizeof(int), false);

				int anim_end = 0;
				spaceConverterServerTcp.recv_data_data((char*)&anim_end, sizeof(int), false);
				bSpaceClientTcp.send_data_data((char*)&anim_end, sizeof(int), false);

				int s = 0;
				spaceConverterServerTcp.recv_data_data((char*)&s, sizeof(int), false);
				bSpaceClientTcp.send_data_data((char*)&s, sizeof(int), false);

				std::vector<char> particle_data_types(s);
				spaceConverterServerTcp.recv_data_data((char*)particle_data_types.data(), sizeof(char) * s, false);
				bSpaceClientTcp.send_data_data((char*)particle_data_types.data(), sizeof(char) * s, false);

				int ack;
				bSpaceClientTcp.recv_data_data((char*)&ack, sizeof(ack), false);
				spaceConverterServerTcp.send_data_data((char*)&ack, sizeof(ack), false);
				printf("sended: particle and data types\n");
			}

			if (message_type == 2)
			{
				bSpaceClientTcp.recv_data_data((char*)&spaceData.bbox_min[0], sizeof(float) * 3, false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.bbox_max[0], sizeof(float) * 3, false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.bbox_dim, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.grid_transform, sizeof(float), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.particle_type, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.block_name_id, sizeof(int), false);
				//bSpaceClientTcp.recv_data_data((char*)&spaceData.value_convert, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.extracted_type, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.dense_type, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.dense_norm, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.object_size, sizeof(float), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.particle_fix_size, sizeof(float), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.filter_min, sizeof(float), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.filter_max, sizeof(float), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.frame, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.anim_type, sizeof(int), false);
				bSpaceClientTcp.recv_data_data((char*)&spaceData.anim_task_counter, sizeof(int), false);

				spaceConverterServerTcp.send_data_data((char*)&spaceData.bbox_min[0], sizeof(float) * 3, false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.bbox_max[0], sizeof(float) * 3, false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.bbox_dim, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.grid_transform, sizeof(float), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.particle_type, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.block_name_id, sizeof(int), false);
				//spaceConverterServerTcp.send_data_data((char*)&spaceData.value_convert, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.extracted_type, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.dense_type, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.dense_norm, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.object_size, sizeof(float), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.particle_fix_size, sizeof(float), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.filter_min, sizeof(float), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.filter_max, sizeof(float), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.frame, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.anim_type, sizeof(int), false);
				spaceConverterServerTcp.send_data_data((char*)&spaceData.anim_task_counter, sizeof(int), false);

				///////////
				// file type				
				spaceConverterServerTcp.recv_data_data((char*)&file_type, sizeof(file_type), false);
				//if (file_type != FTI_OPENVDB) {
				//	printf("file_type != FTI_NANOVDB");
				//	exit(-1);
				//}

				std::size_t size = 0;
				spaceConverterServerTcp.recv_data_data((char*)&size, sizeof(size), false);
				grid_handle_final.resize(size);

				// file type
				int file_type0 = FTI_NONE;
				bSpaceClientTcp.send_data_data((char*)&file_type0, sizeof(file_type0), false);

				std::size_t size0 = 0; //echo about success, keep local grid
				bSpaceClientTcp.send_data_data((char*)&size0, sizeof(size0), false);

				spaceConverterServerTcp.recv_data_data((char*)grid_handle_final.data(), size, false);

				//skip send nvdb to blender
				//bSpaceClientTcp.send_data_data((char*)grid_handle_final.data(), size, false);

				//resend vdb info
				spaceConverterServerTcp.recv_data_data((char*)&spaceData.min_value_reduced, sizeof(spaceData.min_value_reduced), false);
				bSpaceClientTcp.send_data_data((char*)&spaceData.min_value_reduced, sizeof(spaceData.min_value_reduced), false);

				spaceConverterServerTcp.recv_data_data((char*)&spaceData.max_value_reduced, sizeof(spaceData.max_value_reduced), false);
				bSpaceClientTcp.send_data_data((char*)&spaceData.max_value_reduced, sizeof(spaceData.max_value_reduced), false);

				int frames = 1;
				spaceConverterServerTcp.recv_data_data((char*)&frames, sizeof(frames), false);
				bSpaceClientTcp.send_data_data((char*)&frames, sizeof(frames), false);

				int ack;
				bSpaceClientTcp.recv_data_data((char*)&ack, sizeof(ack), false);
				spaceConverterServerTcp.send_data_data((char*)&ack, sizeof(ack), false);
				printf("sended: nvdb\n");

				fromCL.render_running = false;
			}
		}
		//sem_render.notify(); // Wait for the semaphore to be notified

		bSpaceClientTcp.client_close();
		bSpaceClientTcp.server_close();

		spaceConverterServerTcp.client_close();
		spaceConverterServerTcp.server_close();
	}
}


///////////////////////////////////////MAIN LOOP//////////////////////////////////////////////////////
int main(int ac, char** av)
{
//#ifdef _WIN32
//	// Disable assertion pop-up and allow execution to continue
//	//_CrtSetReportMode(_CRT_ASSERT, 0);
//	_CrtSetReportHook(MyReportHook); // Set custom handler
//#endif
	FromCLSpace fromCL;

	fromCL.parse_args(ac, av);

	std::vector<Options> options(fromCL.anim > 0 ? fromCL.anim : 1);
	 
	// Create and run three threads
	std::thread thread1(bspace_loop, std::ref(fromCL));

	TcpConnection blenderClientTcp;
	//blenderClientTcp.init_sockets_data("localhost", fromCL.port);

	for (int i = 0; i < options.size(); i++) {
		auto& op = options[i];
		session_init(fromCL, op, i);
	}

	blenderClientTcp.init_sockets_data("localhost", fromCL.port);

	while (true) {
		
		if (blenderClientTcp.is_error()) //TODO??
		{
			blenderClientTcp.client_close();
			blenderClientTcp.server_close();
			blenderClientTcp.init_sockets_data("localhost", fromCL.port);
		}
		
		////////////////////////////////////////////////////
		if (grid_handle_final.size() > 0) {
			const char* volume_geom_name = getenv("CYCLES_VOLUME_GEOM");
			const char* volume_attr_name = getenv("CYCLES_VOLUME_ATTR");

			if (volume_geom_name && volume_attr_name) {
				for (auto &op : options) {
					xml_set_volume_to_attr(op.scene, volume_geom_name, volume_attr_name, file_type, grid_handle_final);
				}
			}
		}

		//loader_data.clear();
		//render_loop(isHeadNode, loader_data, loader_radius, loader_lights, hanari, world, workers, blenderClientTcp);
		cyclesphi(ac, av, &blenderClientTcp, fromCL, options);
	}

	for (auto &op : options) {
		session_exit(fromCL, op);
	}

	return 0;
}
