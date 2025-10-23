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

int main(int ac, char** av)
{
	FromCL fromCL;
	fromCL.parse_args(ac, av);

	std::vector<Options> options(fromCL.anim > 0 ? fromCL.anim : 1);

	TcpConnection blenderClientTcp;
	//blenderClientTcp.init_sockets_data("localhost", fromCL.port);

	for (int i = 0; i < options.size(); i++) {
		auto& op = options[i];
		session_init(fromCL, op, i);
	}

	while (true) {

		//if (blenderClientTcp.is_error()) 
		{
			blenderClientTcp.client_close();
			blenderClientTcp.server_close();
			blenderClientTcp.init_sockets_data("localhost", fromCL.port);
		}
		
		cyclesphi(ac, av, &blenderClientTcp, fromCL, options);
	}

	for (auto &op : options) {
		session_exit(fromCL, op);
	}
	
	return 0;
}
