/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_PUGIXML

#  include "graph/node_xml_bin.h"
#  include "graph/node.h"

#  include "util/string.h"
#  include "util/transform.h"
#  include "util/path.h"

CCL_NAMESPACE_BEGIN

bool is_file_open(XMLReader& reader) {
	if (!reader.file || !reader.file->is_open()) {
		return false;
	}

	return true;
}

template<typename T>
static void read_array_from_binary_file(XMLReader& reader, array<T>& data, const char* attr)
{
	if (attr == nullptr) {
		fprintf(stderr,	"read_array_from_binary_file: attr is null.\n");
		return;
	}

	// Move to the offset-th byte from the beginning
	size_t offset = atoll(attr);
	reader.file->seekg(offset, std::ios::beg);

	// Read the size of the vector first
	size_t data_size = 0;
	reader.file->read(reinterpret_cast<char*>(&data_size), sizeof(size_t));

	//if (data_size == 0) {
	//	fprintf(stderr,
	//		"read_array_from_binary_file: Wrong size for attribute \"%s\".\n",
	//		attr);
	//	
	//	return;
	//}

	// Resize the vector to hold the data
	data.resize(data_size);

	// Read the actual float data from the file
	reader.file->read(reinterpret_cast<char*>(data.data()), data_size * sizeof(T));
}

bool xml_is_digit(const std::string& str) {
	if (str.empty()) return false;
	size_t start = 0;

	// Check if all remaining characters are digits
	for (size_t i = start; i < str.length(); i++) {
		if (!std::isdigit(str[i])) return false;
	}

	return true;
}

static bool xml_read_boolean(const char* value)
{
	return string_iequals(value, "true") || (atoi(value) != 0);
}

static const char* xml_write_boolean(bool value)
{
	return (value) ? "true" : "false";
}

template<int VECTOR_SIZE, typename T>
static void xml_read_float_array(T& value, xml_attribute attr)
{
	vector<string> tokens;
	string_split(tokens, attr.value());

	if (tokens.size() % VECTOR_SIZE != 0) {
		fprintf(stderr,
			"Wrong VECTOR_SIZE \"%d\" for attribute \"%s\".\n",
			VECTOR_SIZE,
			attr.value());
		return;
	}

	value.resize(tokens.size() / VECTOR_SIZE);
	for (size_t i = 0; i < value.size(); i++) {
		float* value_float = (float*)&value[i];

		for (size_t j = 0; j < VECTOR_SIZE; j++) {
			value_float[j] = (float)atof(tokens[i * VECTOR_SIZE + j].c_str());
		}
	}
}

bool xml_read_node_socket(XMLReader& reader, Node* node, const xml_node xml_root, const SocketType& socket, xml_attribute attr_name, xml_attribute attr_name_ui, xml_attribute attr)
{
	if (socket.type == SocketType::CLOSURE || socket.type == SocketType::UNDEFINED) {
		return false;
	}
	if (socket.flags & SocketType::INTERNAL) {
		return false;
	}

	//for (xml_node xml_node_socket = xml_root.first_child(); xml_node_socket; xml_node_socket = xml_node_socket.next_sibling()) {    
		//ustring xml_node_name(xml_node_socket.name());

	if (!attr) {
		return false;
	}

	if (attr_name && ustring(attr_name.value()) != socket.name ||
		attr_name_ui && ustring(attr_name_ui.value()) != socket.ui_name
		) {
		return false;
	}

	switch (socket.type) {
	case SocketType::BOOLEAN: {
		node->set(socket, xml_read_boolean(attr.value()));
		break;
	}
	case SocketType::BOOLEAN_ARRAY: {
		//vector<string> tokens;
		//string_split(tokens, attr.value());

		array<bool> value;
		//value.resize(tokens.size());
		//for (size_t i = 0; i < value.size(); i++) {
		//  value[i] = xml_read_boolean(tokens[i].c_str());
		//}
		read_array_from_binary_file(reader, value, attr.value());
		node->set(socket, value);
		break;
	}
	case SocketType::FLOAT: {
		node->set(socket, (float)atof(attr.value()));
		break;
	}
	case SocketType::FLOAT_ARRAY: {
		array<float> value;
		if (is_file_open(reader))
			read_array_from_binary_file(reader, value, attr.value());
		else
			xml_read_float_array<1>(value, attr);
		node->set(socket, value);
		break;
	}
	case SocketType::INT: {
		node->set(socket, (int)atoi(attr.value()));
		break;
	}
	case SocketType::UINT: {
		node->set(socket, (uint)atoi(attr.value()));
		break;
	}
	case SocketType::UINT64: {
		node->set(socket, (uint64_t)strtoull(attr.value(), nullptr, 10));
		break;
	}
	case SocketType::INT_ARRAY: {
		//vector<string> tokens;
		//string_split(tokens, attr.value());

		array<int> value;
		//value.resize(tokens.size());
		//for (size_t i = 0; i < value.size(); i++) {
		//  value[i] = (int)atoi(tokens[i].c_str());
		//}
		read_array_from_binary_file(reader, value, attr.value());
		node->set(socket, value);
		break;
	}
	case SocketType::COLOR:
	case SocketType::VECTOR:
	case SocketType::POINT:
	case SocketType::NORMAL: {
		array<float3> value;
		xml_read_float_array<4>(value, attr);
		if (value.size() == 1) {
			node->set(socket, value[0]);
		}
		break;
	}
	case SocketType::COLOR_ARRAY:
	case SocketType::VECTOR_ARRAY:
	case SocketType::POINT_ARRAY:
	case SocketType::NORMAL_ARRAY: {
		array<float3> value;
		if (is_file_open(reader))
			read_array_from_binary_file(reader, value, attr.value());
		else
			xml_read_float_array<4>(value, attr);
		node->set(socket, value);
		break;
	}
	case SocketType::POINT2: {
		array<float2> value;
		xml_read_float_array<2>(value, attr);
		if (value.size() == 1) {
			node->set(socket, value[0]);
		}
		break;
	}
	case SocketType::POINT2_ARRAY: {
		array<float2> value;
		if (is_file_open(reader))
			read_array_from_binary_file(reader, value, attr.value());
		else
			xml_read_float_array<2>(value, attr);
		node->set(socket, value);
		break;
	}
	case SocketType::STRING: {
		node->set(socket, attr.value());
		break;
	}
	case SocketType::ENUM: {
		ustring value(attr.value());

		bool res = false;
		if (xml_is_digit(value.string())) {
			int ivalue = std::atoi(value.c_str());
			res = socket.enum_values->exists(ivalue);
			if (res)
				node->set(socket, ivalue);
		}
		else {
			res = socket.enum_values->exists(value);
			if (res)
				node->set(socket, value);
		}

		if (!res) {
			fprintf(stderr,
				"SocketType::ENUM: Unknown value \"%s\" for attribute \"%s\".\n",
				value.c_str(),
				socket.ui_name.c_str());
		}
		break;
	}
	case SocketType::STRING_ARRAY: {
		vector<string> tokens;
		string_split(tokens, attr.value());

		array<ustring> value;
		value.resize(tokens.size());
		for (size_t i = 0; i < value.size(); i++) {
			value[i] = ustring(tokens[i]);
		}
		node->set(socket, value);
		break;
	}
	case SocketType::TRANSFORM: {
		array<Transform> value;
		xml_read_float_array<12>(value, attr);
		if (value.size() == 1) {
			node->set(socket, value[0]);
		}
		break;
	}
	case SocketType::TRANSFORM_ARRAY: {
		array<Transform> value;
		if (is_file_open(reader))
			read_array_from_binary_file(reader, value, attr.value());
		else
			xml_read_float_array<12>(value, attr);
		node->set(socket, value);
		break;
	}
	case SocketType::NODE: {
		ustring value(attr.value());
		map<ustring, Node*>::iterator it = reader.node_map.find(value);
		if (it != reader.node_map.end()) {
			Node* value_node = it->second;
			//if (value_node->is_a(socket.node_type)) 
			{
				node->set(socket, value_node);
			}
		}
		break;
	}
	case SocketType::NODE_ARRAY: {
		vector<string> tokens;
		string_split(tokens, attr.value());

		array<Node*> value;
		value.resize(tokens.size());
		for (size_t i = 0; i < value.size(); i++) {
			map<ustring, Node*>::iterator it = reader.node_map.find(ustring(tokens[i]));
			if (it != reader.node_map.end()) {
				Node* value_node = it->second;
				//value[i] = (value_node->is_a(socket.node_type)) ? value_node : NULL;
				value[i] = value_node;
			}
			else {
				value[i] = NULL;
			}
		}

		//check
		bool null_values = false;
		for (size_t i = 0; i < value.size(); i++) {
			if (value[i] == NULL) {
				fprintf(stderr,
					"SocketType::NODE_ARRAY: Node \"%s\" was not found in attribute \"%s\".\n",
					tokens[i].c_str(),
					socket.ui_name.c_str());

				null_values = true;
			}
		}
		if (!null_values)
			node->set(socket, value);

		break;
	}
	case SocketType::CLOSURE:
	case SocketType::UNDEFINED:
	case SocketType::NUM_TYPES:
		break;
	//default:
	//	return false;
	}
	//}

	//socket_found = true;
	//break;
	return true;
}


void xml_read_node(XMLReader& reader, Node* node, const xml_node xml_root)
{
	xml_attribute name_attr = xml_root.attribute("name");
	if (name_attr) {
		node->name = ustring(name_attr.value());
	}

	for (xml_node xml_node_socket : xml_root.children("socket")) {
		bool socket_found = false;

		xml_attribute attr_name = xml_node_socket.attribute("name");
		xml_attribute attr_name_ui = xml_node_socket.attribute("ui_name");	
		xml_attribute attr = xml_node_socket.attribute("value");

		for(const SocketType & socket: node->type->inputs) {
			socket_found |= xml_read_node_socket(reader, node, xml_root, socket, attr_name, attr_name_ui, attr);

			if (socket_found)
				break;
		}

		if (false /*!socket_found*/) {
			for(const SocketType & socket: node->type->outputs) {
				socket_found |= xml_read_node_socket(reader, node, xml_root, socket, attr_name, attr_name_ui, attr);

				if (socket_found)
					break;
			}
		}

		if (!socket_found) {
			fprintf(stderr,
				"Unknown attribute %s in \"%s\".\n",
				(attr) ? attr.value() : "empty",
				node->name.c_str());
		}
	}

	if (!node->name.empty()) {
		reader.node_map[node->name] = node;
	}
}

xml_node xml_write_node(Node* node, xml_node xml_root)
{
	xml_node xml_node = xml_root.append_child(node->type->name.c_str());

	xml_node.append_attribute("name") = node->name.c_str();

	for(const SocketType & socket: node->type->inputs) {
		if (socket.type == SocketType::CLOSURE || socket.type == SocketType::UNDEFINED) {
			continue;
		}
		if (socket.flags & SocketType::INTERNAL) {
			continue;
		}
		if (node->has_default_value(socket)) {
			continue;
		}

		xml_attribute attr = xml_node.append_attribute(socket.ui_name.c_str());

		switch (socket.type) {
		case SocketType::BOOLEAN: {
			attr = xml_write_boolean(node->get_bool(socket));
			break;
		}
		case SocketType::BOOLEAN_ARRAY: {
			std::stringstream ss;
			const array<bool>& value = node->get_bool_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				ss << xml_write_boolean(value[i]);
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::FLOAT: {
			attr = (double)node->get_float(socket);
			break;
		}
		case SocketType::FLOAT_ARRAY: {
			std::stringstream ss;
			const array<float>& value = node->get_float_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				ss << value[i];
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::INT: {
			attr = node->get_int(socket);
			break;
		}
		case SocketType::UINT: {
			attr = node->get_uint(socket);
			break;
		}
		case SocketType::UINT64: {
			attr = node->get_uint64(socket);
			break;
		}
		case SocketType::INT_ARRAY: {
			std::stringstream ss;
			const array<int>& value = node->get_int_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				ss << value[i];
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::COLOR:
		case SocketType::VECTOR:
		case SocketType::POINT:
		case SocketType::NORMAL: {
			float3 value = node->get_float3(socket);
			attr =
				string_printf("%g %g %g", (double)value.x, (double)value.y, (double)value.z).c_str();
			break;
		}
		case SocketType::COLOR_ARRAY:
		case SocketType::VECTOR_ARRAY:
		case SocketType::POINT_ARRAY:
		case SocketType::NORMAL_ARRAY: {
			std::stringstream ss;
			const array<float3>& value = node->get_float3_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				ss << string_printf(
					"%g %g %g", (double)value[i].x, (double)value[i].y, (double)value[i].z);
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::POINT2: {
			float2 value = node->get_float2(socket);
			attr = string_printf("%g %g", (double)value.x, (double)value.y).c_str();
			break;
		}
		case SocketType::POINT2_ARRAY: {
			std::stringstream ss;
			const array<float2>& value = node->get_float2_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				ss << string_printf("%g %g", (double)value[i].x, (double)value[i].y);
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::STRING:
		case SocketType::ENUM: {
			attr = node->get_string(socket).c_str();
			break;
		}
		case SocketType::STRING_ARRAY: {
			std::stringstream ss;
			const array<ustring>& value = node->get_string_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				ss << value[i];
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::TRANSFORM: {
			Transform tfm = node->get_transform(socket);
			std::stringstream ss;
			for (int i = 0; i < 3; i++) {
				ss << string_printf("%g %g %g %g ",
					(double)tfm[i][0],
					(double)tfm[i][1],
					(double)tfm[i][2],
					(double)tfm[i][3]);
			}
			ss << string_printf("%g %g %g %g", 0.0, 0.0, 0.0, 1.0);
			attr = ss.str().c_str();
			break;
		}
		case SocketType::TRANSFORM_ARRAY: {
			std::stringstream ss;
			const array<Transform>& value = node->get_transform_array(socket);
			for (size_t j = 0; j < value.size(); j++) {
				const Transform& tfm = value[j];

				for (int i = 0; i < 3; i++) {
					ss << string_printf("%g %g %g %g ",
						(double)tfm[i][0],
						(double)tfm[i][1],
						(double)tfm[i][2],
						(double)tfm[i][3]);
				}
				ss << string_printf("%g %g %g %g", 0.0, 0.0, 0.0, 1.0);
				if (j != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::NODE: {
			Node* value = node->get_node(socket);
			if (value) {
				attr = value->name.c_str();
			}
			break;
		}
		case SocketType::NODE_ARRAY: {
			std::stringstream ss;
			const array<Node*>& value = node->get_node_array(socket);
			for (size_t i = 0; i < value.size(); i++) {
				if (value[i]) {
					ss << value[i]->name.c_str();
				}
				if (i != value.size() - 1) {
					ss << " ";
				}
			}
			attr = ss.str().c_str();
			break;
		}
		case SocketType::CLOSURE:
		case SocketType::UNDEFINED:
		case SocketType::NUM_TYPES:
			break;
		}
	}

	return xml_node;
}

CCL_NAMESPACE_END

#endif /* WITH_PUGIXML */
