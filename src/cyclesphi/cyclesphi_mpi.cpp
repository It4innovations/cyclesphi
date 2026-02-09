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
#include <mpi.h>

class MpiConnection : public TcpConnection {
public:
	MpiConnection() : TcpConnection() {
	}

	void init_mpi(int argc, char** argv)
	{
		//int provided = 0;
		// MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
		MPI_Init(&argc, &argv);

		// Get the number of processes
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		// world_size = 2;

		// Get the rank of the process
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		// world_rank = 0;

		// Get the name of the processor
		char processor_name[MPI_MAX_PROCESSOR_NAME];
		int name_len;
		MPI_Get_processor_name(processor_name, &name_len);

		if (world_rank < 2) {
			// Print off a hello world message
			printf(
				"Start from processor %s, rank %d"
				" out of %d processors\n",
				processor_name,
				world_rank,
				world_size);

			fflush(0);

			// mpi_print_memory(world_rank);
		}

	}

	void close_mpi() 
	{
		if (world_rank < 2) {
			printf(
				"End from processor, rank %d"
				" out of %d processors\n",
				world_rank,
				world_size);

			fflush(0);
		}

		MPI_Finalize();
	}

	void init_sockets_data(const char* server = NULL, int port = 0, bool is_server = true) override
	{
		if (world_rank == 0)
			TcpConnection::init_sockets_data(server, port);
	}
	void client_close() override
	{
		if (world_rank == 0)
			TcpConnection::client_close();
	}
	void server_close() override
	{
		if (world_rank == 0)
			TcpConnection::server_close();
	}

	bool is_error() override
	{
		if (world_rank == 0)
			TcpConnection::is_error();

		return false;
	}

	void send_data_data(char *data, size_t size, char ack = 0) override
	{
		if (world_size > 1 && frame >=0 && frame < world_size) {
			if (world_rank > 0 && world_rank == frame)
				MPI_Send(data, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
			else if (world_rank == 0)
				MPI_Recv(data, size, MPI_CHAR, frame, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if (world_rank == 0)
			TcpConnection::send_data_data(data, size, ack);
	}

	void recv_data_data(char *data, size_t size, char ack = 0) override
	{
		if (world_rank == 0)
			TcpConnection::recv_data_data(data, size, ack);

		MPI_Bcast(data, size, MPI_CHAR, 0, MPI_COMM_WORLD);
	}

	void send_gpujpeg(char* dmem, char* pixels, int width, int height, int format) override
	{
#ifdef WITH_CLIENT_GPUJPEG
		size_t frame_size = 0;
		TcpConnection::gpujpeg_encode(width, height, format, (uint8_t*)dmem, (uint8_t*)pixels, frame_size);

		//if (world_rank == 0) {
		//TcpConnection::send_data_data((char*)&frame_size, sizeof(int));
		//TcpConnection::send_data_data((char*)g_image_compressed, frame_size);
		//}

		if (world_size > 1 && frame >= 0 && frame < world_size) {
			if (world_rank > 0 && world_rank == frame) {
				MPI_Send((char*)&frame_size, sizeof(int), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
				MPI_Send((char*)g_image_compressed, frame_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
			}
			else if (world_rank == 0) {
				MPI_Recv((char*)&frame_size, sizeof(int), MPI_CHAR, frame, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv((char*)g_image_compressed, frame_size, MPI_CHAR, frame, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}


		if (world_rank == 0) {
			TcpConnection::send_data_data((char*)&frame_size, sizeof(int));
			TcpConnection::send_data_data((char*)g_image_compressed, frame_size);
		}
#endif
	}

	void recv_gpujpeg(char* dmem, char* pixels, int width, int height, int format) override
	{
		if (world_rank == 0)
			TcpConnection::recv_gpujpeg(dmem, pixels, width, height, format);
	}

	int get_world_rank() const
	{
		return world_rank;
	}

	int get_world_size() const
	{
		return world_size;
	}

private: 	
	int world_rank = 0;
	int world_size = 1;
};

int main(int argc, char** argv)
{
#if 0
	int start = 0;
	while (!start) {
		Sleep(1000);
	}
#endif

	FromCL fromCL;
	fromCL.parse_args(argc, argv);

	MpiConnection blenderClientTcp;
	blenderClientTcp.init_mpi(argc, argv);

	fromCL.use_mpi = true;
	fromCL.world_rank = blenderClientTcp.get_world_rank();
	fromCL.world_size = blenderClientTcp.get_world_size();

	std::vector<Options> options(fromCL.anim > 0 ? fromCL.anim : 1);	

	for (int i = 0; i < options.size(); i++) {
		auto& op = options[i];
		session_init(fromCL, op, i);
	}

	//blenderClientTcp.init_sockets_data("localhost", fromCL.port);

	while (true) {
		//if (blenderClientTcp.is_error()) 
		{
			blenderClientTcp.client_close();
			blenderClientTcp.server_close();
			blenderClientTcp.init_sockets_data("localhost", fromCL.port);
		}
		
		cyclesphi(argc, argv, &blenderClientTcp, fromCL, options);
	}

	for (auto &op : options) {
		session_exit(fromCL, op);
	}

	blenderClientTcp.client_close();
	blenderClientTcp.server_close();

	blenderClientTcp.close_mpi();
	
	return 0;
}
