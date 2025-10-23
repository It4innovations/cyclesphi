/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstdio>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <regex>

#include "graph/node_xml_bin.h"

#include "scene/alembic.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/object.h"
#include "scene/osl.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "scene/mesh.h"
#include "scene/hair.h"
#include "scene/volume.h"
#include "scene/pointcloud.h"
#include "scene/image_vdb.h"

#include "subd/patch.h"
#include "subd/split.h"

#include "util/path.h"
#include "util/projection.h"
#include "util/transform.h"
#include "util/xml.h"

#include "app/cycles_xml_bin.h"

#include <openvdb/io/Stream.h>
#include <nanovdb/io/IO.h>

#include "scene/image_oiio.h"

#define READ_ATTR_I(name,type) \
    const xml_attribute attr_##name = node_attribute.attribute(#name); \
    type name = (type) 0; \
    if (attr_##name) \
        name = (type) atoi(attr_##name.value());

#define READ_ATTR_ULL(name,type) \
    const xml_attribute attr_##name = node_attribute.attribute(#name); \
    type name = (type) 0; \
    if (attr_##name) \
        name = (type) atoll(attr_##name.value());

#define READ_ATTR_STR(name,type) \
    const xml_attribute attr_##name = node_attribute.attribute(#name); \
    type name; \
    if (attr_##name) \
        name = (type) (attr_##name.value());

CCL_NAMESPACE_BEGIN

/* XML reading state */

struct XMLReadState : public XMLReader {
	Scene* scene;      /* Scene pointer. */

	XMLReadState() :
		scene(nullptr)
	{
	}
};

template<typename T>
static void read_vector_from_binary_file(XMLReader& reader, vector<T>& data, const char* attr)
{
	if (attr == nullptr) {
		fprintf(stderr, "read_vector_from_binary_file: attr is null.\n");
		return;
	}

	// Move to the offset-th byte from the beginning
	size_t offset = atoll(attr);
	reader.file->seekg(offset, std::ios::beg);

	// Read the size of the vector first
	size_t data_size = 0;
	reader.file->read(reinterpret_cast<char*>(&data_size), sizeof(size_t));

	if (data_size == 0) {
		fprintf(stderr,
			"read_vector_from_binary_file: Wrong size for attribute \"%s\".\n",
			attr);

		return;
	}

	// Resize the vector to hold the data
	data.resize(data_size);

	// Read the actual float data from the file
	reader.file->read(reinterpret_cast<char*>(data.data()), data_size * sizeof(T));
}


/* Attribute Reading */

static bool xml_read_int(int* value, const xml_node node, const char* name)
{
	const xml_attribute attr = node.attribute(name);

	if (attr) {
		*value = atoi(attr.value());
		return true;
	}

	return false;
}

static bool xml_read_int_array(vector<int>& value, const xml_node node, const char* name)
{
	const xml_attribute attr = node.attribute(name);

	if (attr) {
		vector<string> tokens;
		string_split(tokens, attr.value());

		for (const string &token : tokens) {
			value.push_back(atoi(token.c_str()));
		}

		return true;
	}

	return false;
}

static bool xml_read_float(float* value, const xml_node node, const char* name)
{
	const xml_attribute attr = node.attribute(name);

	if (attr) {
		*value = (float)atof(attr.value());
		return true;
	}

	return false;
}

static bool xml_read_float_array(vector<float>& value, const xml_node node, const char* name)
{
	const xml_attribute attr = node.attribute(name);

	if (attr) {
		vector<string> tokens;
		string_split(tokens, attr.value());

        for (const string &token : tokens) {
			value.push_back((float)atof(token.c_str()));
		}

		return true;
	}

	return false;
}

static bool xml_read_float3(float3* value, const xml_node node, const char* name)
{
	vector<float> array;

	if (xml_read_float_array(array, node, name) && array.size() == 3) {
		*value = make_float3(array[0], array[1], array[2]);
		return true;
	}

	return false;
}

static bool xml_read_float3_array(vector<float3>& value, const xml_node node, const char* name)
{
	vector<float> array;

	if (xml_read_float_array(array, node, name)) {
		for (size_t i = 0; i < array.size(); i += 3) {
			value.push_back(make_float3(array[i + 0], array[i + 1], array[i + 2]));
		}

		return true;
	}

	return false;
}

static bool xml_read_float4(float4* value, const xml_node node, const char* name)
{
	vector<float> array;

	if (xml_read_float_array(array, node, name) && array.size() == 4) {
		*value = make_float4(array[0], array[1], array[2], array[3]);
		return true;
	}

	return false;
}

static bool xml_read_string(string* str, const xml_node node, const char* name)
{
	const xml_attribute attr = node.attribute(name);

	if (attr) {
		*str = attr.value();
		return true;
	}

	return false;
}

static bool xml_equal_string(const xml_node node, const char* name, const char* value)
{
	const xml_attribute attr = node.attribute(name);

	if (attr) {
		return string_iequals(attr.value(), value);
	}

	return false;
}

/* Camera */

static void xml_read_camera(XMLReadState& state, const xml_node node, Camera* cam)
{
	//Camera* cam = state.scene->camera;

	xml_read_node(state, cam, node);

	cam->need_flags_update = true;
	cam->update(state.scene);
}


/* Shader */

static void xml_read_shader_graph(XMLReadState& state, Shader* shader, const xml_node graph_node)
{
	xml_read_node(state, shader, graph_node);

	unique_ptr<ShaderGraph> graph = make_unique<ShaderGraph>();

	/* local state, shader nodes can't link to nodes outside the shader graph */
	XMLReader graph_reader;
	graph_reader.node_map[ustring("output")] = graph->output();
	graph_reader.node_map[ustring("Material Output")] = graph->output();
	graph_reader.file = state.file;

	for (const xml_node node : graph_node.children()) {
		ustring node_name(node.name());

		if (node_name == "socket") {
			continue; //skip
		}

		if (node_name == "connect") {
			/* connect nodes */

			const xml_attribute xml_attr_from_node = node.attribute("from_node");
			const xml_attribute xml_attr_from_socket = node.attribute("from_socket");
			const xml_attribute xml_attr_to_node = node.attribute("to_node");
			const xml_attribute xml_attr_to_socket = node.attribute("to_socket");

			if (xml_attr_from_node && xml_attr_to_node) {
				ustring from_node_name(node.attribute("from_node").value());
				//ustring from_socket_name(node.attribute("from_socket").value());
				ustring to_node_name(node.attribute("to_node").value());
				//ustring to_socket_name(node.attribute("to_socket").value());

				/* find nodes and sockets */
				ShaderOutput* output = nullptr;
				ShaderInput* input = nullptr;

				if (graph_reader.node_map.find(from_node_name) != graph_reader.node_map.end()) {
					ShaderNode* fromnode = (ShaderNode*)graph_reader.node_map[from_node_name];

					const xml_attribute from_socket = node.attribute("from_socket");
					const xml_attribute from_socket_ui = node.attribute("from_socket_ui");

					for(ShaderOutput * out: fromnode->outputs) {
						//if (string_iequals(out->socket_type.ui_name.string(), from_socket_name.string()))

						if (from_socket && ustring(from_socket.value()) != out->socket_type.name ||
							from_socket_ui && ustring(from_socket_ui.value()) != out->socket_type.ui_name
							) {
							continue;
						}

						output = out;
					}

					if (!output) {
						fprintf(stderr,
							"Unknown output socket name \"%s\".\n",
							from_node_name.c_str());
					}
				}
				else {
					fprintf(stderr, "Unknown shader node name \"%s\".\n", from_node_name.c_str());
				}

				if (graph_reader.node_map.find(to_node_name) != graph_reader.node_map.end()) {
					ShaderNode* tonode = (ShaderNode*)graph_reader.node_map[to_node_name];

					const xml_attribute to_socket = node.attribute("to_socket");
					const xml_attribute to_socket_ui = node.attribute("to_socket_ui");

					for(ShaderInput * in: tonode->inputs) {
						//if (string_iequals(in->socket_type.ui_name.string(), to_socket_name.string())) {

						if (to_socket && ustring(to_socket.value()) != in->socket_type.name ||
							to_socket_ui && ustring(to_socket_ui.value()) != in->socket_type.ui_name
							) {
							continue;
						}

						input = in;
					}

					if (!input) {
						fprintf(stderr,
							"Unknown input socket name \"%s\".\n", to_node_name.c_str());
					}
				}
				else {
					fprintf(stderr, "Unknown shader node name \"%s\".\n", to_node_name.c_str());
				}

				/* connect */
				if (output && input) {
					graph->connect(output, input);
				}
			}
			else {
				fprintf(stderr, "Invalid from or to value for connect node.\n");
			}
		}
		else {

			ShaderNode* snode = nullptr;

			if (node_name == "background") {
				node_name = "background_shader";
			}

			const NodeType* node_type = NodeType::find(node_name);

      if (!node_type) {
        fprintf(stderr, "Unknown shader node \"%s\".\n", node.name());
        continue;
			}
			else if (node_type->type != NodeType::SHADER) {
				fprintf(stderr, "Node type \"%s\" is not a shader node.\n", node_type->name.c_str());
				continue;
			}
			else if (node_type->create == nullptr) {
				fprintf(stderr, "Can't create abstract node type \"%s\".\n", node_type->name.c_str());
				continue;
			}

			// snode = (ShaderNode*)node_type->create(node_type);
			// snode->set_owner(graph);
			snode = graph->create_node(node_type);
			//snode->set_owner(graph.get());
			//}			

			xml_read_node(graph_reader, snode, node);

			if (node_name == "image_texture") {
				ImageTextureNode* img = (ImageTextureNode*)snode;
				//ustring filename(path_join(state.base, img->get_filename().string()));
				//img->set_filename(filename);

				std::string filename = img->get_filename().string();

				if (xml_is_digit(filename) && img->handle.empty()) {
					ImageManager* image_manager = state.scene->image_manager.get();

					vector<char> image_buffer;
					read_vector_from_binary_file(state, image_buffer, filename.c_str());

					////////////////////////////
					// MetaData
					const xml_node node_attribute = node;
					ImageMetaData attr;// = img->handle.metadata();
					//int channels;			
					READ_ATTR_I(channels, int);
					attr.channels = channels;

					//size_t width, height, depth;
					READ_ATTR_ULL(width, size_t);
					attr.width = width;
					READ_ATTR_ULL(height, size_t);
					attr.height = height;
					READ_ATTR_ULL(depth, size_t);
					attr.depth = depth;
					//size_t byte_size;
					//READ_ATTR_ULL(byte_size, size_t);
					//attr.byte_size = byte_size;
					//ImageDataType type;
					READ_ATTR_I(type, int);
					attr.type = (ImageDataType)type;

					///* Optional color space, defaults to raw. */
					//ustring colorspace;
					READ_ATTR_STR(colorspace, ustring);
					attr.colorspace = colorspace;
					//string colorspace_file_hint;
					READ_ATTR_STR(colorspace_file_hint, string);
					attr.colorspace_file_hint = colorspace_file_hint;
					//const char* colorspace_file_format;
					READ_ATTR_STR(colorspace_file_format, string);
					attr.colorspace_file_format = colorspace_file_format.c_str();

					///* Optional transform for 3D images. */
					//bool use_transform_3d;
					//Transform transform_3d;

					///* Automatically set. */
					//bool compress_as_srgb;					
					////////////////////////////
					unique_ptr<ImageLoader> loader = make_unique<OIIOImageLoader>(filename, image_buffer, attr);
					ImageParams params = img->image_params();
					img->handle = image_manager->add_image(std::move(loader), params, false);
				}
			} 
			else if(node_name == "environment_texture") {
				EnvironmentTextureNode* img = (EnvironmentTextureNode*)snode;
				//ustring filename(path_join(state.base, img->get_filename().string()));
				//img->set_filename(filename);

				std::string filename = img->get_filename().string();

				if (xml_is_digit(filename) && img->handle.empty()) {
					ImageManager* image_manager = state.scene->image_manager.get();

					vector<char> image_buffer;
					read_vector_from_binary_file(state, image_buffer, filename.c_str());
					////////////////////////////
					// MetaData
					const xml_node node_attribute = node;
					ImageMetaData attr;// = img->handle.metadata();
					//int channels;			
					READ_ATTR_I(channels, int);
					attr.channels = channels;

					//size_t width, height, depth;
					READ_ATTR_ULL(width, size_t);
					attr.width = width;
					READ_ATTR_ULL(height, size_t);
					attr.height = height;
					READ_ATTR_ULL(depth, size_t);
					attr.depth = depth;
					//size_t byte_size;
					//READ_ATTR_ULL(byte_size, size_t);
					//attr.byte_size = byte_size;
					//ImageDataType type;
					READ_ATTR_I(type, int);
					attr.type = (ImageDataType)type;

					///* Optional color space, defaults to raw. */
					//ustring colorspace;
					READ_ATTR_STR(colorspace, ustring);
					attr.colorspace = colorspace;
					//string colorspace_file_hint;
					READ_ATTR_STR(colorspace_file_hint, string);
					attr.colorspace_file_hint = colorspace_file_hint;
					//const char* colorspace_file_format;
					READ_ATTR_STR(colorspace_file_format, string);
					attr.colorspace_file_format = colorspace_file_format.c_str();

					///* Optional transform for 3D images. */
					//bool use_transform_3d;
					//Transform transform_3d;

					///* Automatically set. */
					//bool compress_as_srgb;					
					////////////////////////////
					unique_ptr<ImageLoader> loader = make_unique<OIIOImageLoader>(filename, image_buffer, attr);
					ImageParams params = img->image_params();
					img->handle = image_manager->add_image(std::move(loader), params, false);
				}
			}

			// if (snode) {
			// 	/* add to graph */
			// 	graph->add_node(snode);
			// }
		}
	}

	//shader->set_graph(graph);
	shader->set_graph(std::move(graph));
	shader->tag_update(state.scene);
}

static void xml_read_shader(XMLReadState& state, const xml_node node)
{
	const xml_attribute name_attr = node.attribute("name");
	if (name_attr && string_iequals(name_attr.value(), "default_background")) {
		Shader* shader = state.scene->default_background;
		xml_read_shader_graph(state, shader, node);
	}
	else if (name_attr && string_iequals(name_attr.value(), "default_empty")) {
		Shader* shader = state.scene->default_empty;
		xml_read_shader_graph(state, shader, node);
	}
	else if (name_attr && string_iequals(name_attr.value(), "default_light")) {
		Shader* shader = state.scene->default_light;
		xml_read_shader_graph(state, shader, node);
	}
	else if (name_attr && string_iequals(name_attr.value(), "default_surface")) {
		Shader* shader = state.scene->default_surface;
		xml_read_shader_graph(state, shader, node);
	}
	else if (name_attr && string_iequals(name_attr.value(), "default_volume")) {
		Shader* shader = state.scene->default_volume;
		xml_read_shader_graph(state, shader, node);
	}
	else 
	{
		Shader* shader = state.scene->create_node<Shader>();
		xml_read_shader_graph(state, shader, node);

	}
}

/* Background */

static void xml_read_background(XMLReadState& state, const xml_node node)
{
	/* Background Settings */
	xml_read_node(state, state.scene->background, node);
}

/* Mesh */

static void xml_read_geom(XMLReadState& state, const xml_node xml_node_geom)
{
	const xml_attribute attr_gt = xml_node_geom.attribute("geometry_type");
	if (!attr_gt) {
		fprintf(stderr, "Missing geometry type in %s\n", xml_node_geom.value());
		return;
	}
	Geometry::Type geometry_type = (Geometry::Type)atoi(attr_gt.value());

	Geometry* geom = nullptr;
	switch (geometry_type) {
	case Geometry::Type::MESH:
		geom = state.scene->create_node<Mesh>();
		break;
	case Geometry::Type::HAIR:
		geom = state.scene->create_node<Hair>();
		break;
	case Geometry::Type::VOLUME:
		geom = state.scene->create_node<Volume>();
		break;
	case Geometry::Type::POINTCLOUD:
		geom = state.scene->create_node<PointCloud>();
		break;
	case Geometry::Type::LIGHT:
		geom = state.scene->create_node<Light>();
		break;
	}

	xml_read_node(state, geom, xml_node_geom);

	for (const xml_node node_attribute : xml_node_geom.children("attribute")) {
		ustring name("");
		const xml_attribute attr_name = node_attribute.attribute("name");
		if (attr_name)
			name = attr_name.value();

		READ_ATTR_I(std, AttributeStandard);

		TypeDesc type;

		READ_ATTR_I(basetype, unsigned char);      ///< C data type at the heart of our type
		READ_ATTR_I(aggregate, unsigned char);     ///< What kind of AGGREGATE is it?
		READ_ATTR_I(vecsemantics, unsigned char);  ///< Hint: What does the aggregate represent?
		READ_ATTR_I(reserved, unsigned char);      ///< Reserved for future expansion
		READ_ATTR_I(arraylen, int);      ///< Array length, 0 = not array, -1 = unsized

		type.basetype = basetype;
		type.aggregate = aggregate;
		type.vecsemantics = vecsemantics;
		type.reserved = reserved;
		type.arraylen = arraylen;

		READ_ATTR_I(element, AttributeElement);
		READ_ATTR_I(flags, uint);

		Attribute* attr = geom->attributes.add(name, type, element);
		attr->std = std;
		attr->flags = flags;

		const xml_attribute attr_buffer = node_attribute.attribute("buffer");
		if (attr_buffer) {

			const xml_attribute attr_volume_type = node_attribute.attribute("volume_type");
			if (attr_volume_type) {
				ustring volume_type(attr_volume_type.value());
				if (volume_type == "openvdb") {
					//std::stringstream ss;
					//ss << attr_buffer;
					//std::string str = ss.str();
					//std::replace(str.begin(), str.end(), ' ', '_');

					openvdb::GridBase::Ptr grid;
					std::string filename = attr_buffer.value();

					openvdb::initialize();

					if (xml_is_digit(filename)) {
						//string filename(path_join(state.base, attr_buffer.value()));

						//openvdb::initialize();

						vector<char> file_content;
						read_vector_from_binary_file(state, file_content, filename.c_str());

						// Convert the vector<char> to a stringstream
						std::istringstream stream(std::string(file_content.begin(), file_content.end()), std::ios_base::binary);
						// Create a VDB input stream from the stringstream
						openvdb::io::Stream vdbStream(stream);
						// Read the grid from the stream
						openvdb::GridPtrVecPtr grids = vdbStream.getGrids();
						
						// Find the first FloatGrid in the grids vector and return it
						for (auto& g : *grids) {
							if (g->isType<openvdb::FloatGrid>() && g->getName() == name.string()) {
								grid = openvdb::gridPtrCast<openvdb::FloatGrid>(g);
								break;
							}
						}
					}
					else {
						//grid = openvdb::io::File(filename).readGrid(name.c_str());
						openvdb::io::File vdbFile(filename);

						vdbFile.open(); // Explicitly open the file
						std::string n = name.string();
						grid = vdbFile.readGrid(n); // Pass `name` directly if it is std::string
						vdbFile.close(); // Close the file after reading
					}
					//attr->data_voxel() = state.scene->image_manager->add_image(loader, params, false);

					unique_ptr<ImageLoader> loader = make_unique<VDBImageLoader>(grid, name.string());
					const ImageParams params;
					attr->data_voxel() = state.scene->image_manager->add_image(std::move(loader), params);

				}
				else if (volume_type == "nanovdb") {
					//nanovdb::NanoGrid<float>* nanogrid = nullptr;
					//size_t nanogrid_size = 0;
					vector<char> nanogrid;
					std::string filename = attr_buffer.value();

					if (xml_is_digit(filename)) {
						//
						read_vector_from_binary_file(state, nanogrid, filename.c_str());
					}
					else {
						nanovdb::GridHandle<nanovdb::HostBuffer> grid_handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(filename);
						size_t nanogrid_size = grid_handle.size();
						nanogrid.resize(nanogrid_size);
						memcpy(nanogrid.data(), grid_handle.data(), nanogrid_size);						
					}
					unique_ptr<ImageLoader> loader = make_unique<NanoVDBImageLoader>(nanogrid);
					const ImageParams params;
					attr->data_voxel() = state.scene->image_manager->add_image(std::move(loader), params, false);
				}
				else if (volume_type == "raw") {
					vector<char> raw_data;
					std::string filename = attr_buffer.value();

					const xml_attribute attr_raw_width = node_attribute.attribute("raw_dx");
					if (!attr_raw_width) {
						std::cerr << "Error: missing raw_dx" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_height = node_attribute.attribute("raw_dy");
					if (!attr_raw_height) {
						std::cerr << "Error: missing raw_dy" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_depth = node_attribute.attribute("raw_dz");
					if (!attr_raw_depth) {
						std::cerr << "Error: missing raw_dz" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_scal_x = node_attribute.attribute("scal_x");
					if (!attr_raw_scal_x) {
						std::cerr << "Error: missing scal_x" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_scal_y = node_attribute.attribute("scal_y");
					if (!attr_raw_scal_y) {
						std::cerr << "Error: missing scal_y" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_scal_z = node_attribute.attribute("scal_z");
					if (!attr_raw_scal_z) {
						std::cerr << "Error: missing scal_z" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_type = node_attribute.attribute("raw_type");
					if (!attr_raw_type) {
						std::cerr << "Error: missing raw_type" << filename << std::endl;
						continue;
					}

					const xml_attribute attr_raw_channels = node_attribute.attribute("raw_channels");
					if (!attr_raw_channels) {
						std::cerr << "Error: missing raw_channels" << filename << std::endl;
						continue;
					}
										
					int width = std::stoi(attr_raw_width.value());
					int height = std::stoi(attr_raw_height.value());
					int depth = std::stoi(attr_raw_depth.value());

					float scal_x = std::stof(attr_raw_scal_x.value());
					float scal_y = std::stof(attr_raw_scal_y.value());
					float scal_z = std::stof(attr_raw_scal_z.value());

					std::string sraw_type(attr_raw_type.value());
					int channels = std::stoi(attr_raw_channels.value());

					RAWImageLoader::RAWImageLoaderType type = RAWImageLoader::RAWImageLoaderType::eRawFloat;
					if (sraw_type == "byte")
						type = RAWImageLoader::RAWImageLoaderType::eRawByte;
					else if (sraw_type == "half")
						type = RAWImageLoader::RAWImageLoaderType::eRawHalf;
					else if (sraw_type == "ushort")
						type = RAWImageLoader::RAWImageLoaderType::eRawUShort;

					if (xml_is_digit(filename)) {
						read_vector_from_binary_file(state, raw_data, filename.c_str());
					}
					else {
						// Open file in binary mode and move pointer to end to get file size
						std::ifstream file(filename, std::ios::binary | std::ios::ate);

						if (!file) {
							std::cerr << "Error: Could not open file " << filename << std::endl;
							continue;
						}

						// Get file size
						std::streamsize size = file.tellg();
						file.seekg(0, std::ios::beg);

						// Allocate buffer and read file into it
						raw_data.resize(size);
						if (!file.read(raw_data.data(), size)) {
							std::cerr << "Error reading file!" << std::endl;
							continue;
						}

						file.close();
					}

					unique_ptr<ImageLoader> loader = make_unique<RAWImageLoader>(raw_data, width, height, depth, scal_x, scal_y, scal_z, type, channels);
					const ImageParams params;
					attr->data_voxel() = state.scene->image_manager->add_image(std::move(loader), params, false);
				}
			}
			else {
				std::string filename = attr_buffer.value();
				if (xml_is_digit(filename)) {
					read_vector_from_binary_file(state, attr->buffer, filename.c_str());
				}
				else {
					fprintf(stderr, "attr_volume_type is empty\n");
				}
			}
		}
	}
}

/* Light */

//static void xml_read_light(XMLReadState& state, const xml_node node)
//{
//	Light* light = state.scene->create_node<Light>();
//
//	xml_read_node(state, light, node);
//}
//
//static void xml_read_particle_system(XMLReadState& state, const xml_node node)
//{
//	ParticleSystem* ps = state.scene->create_node<ParticleSystem>();
//
//	xml_read_node(state, ps, node);
//}
//
//static void xml_read_pass(XMLReadState& state, const xml_node node)
//{
//	Pass* pass = state.scene->create_node<Pass>();
//
//	xml_read_node(state, pass, node);
//}
//
//static void xml_read_procedural(XMLReadState& state, const xml_node node)
//{
//	//TODO
//	//Procedural* procedural = new Procedural();
//	//xml_read_node(state, procedural, node);
//	//state.scene->procedurals.push_back(procedural);
//}

/* Object */

static void xml_read_object(XMLReadState& state, const xml_node xml_node_obj)
{
	Scene* scene = state.scene;

	Object* object = state.scene->create_node<Object>();

	xml_read_node(state, object, xml_node_obj);

	for (const xml_node node_attribute : xml_node_obj.children("attribute")) {
		ustring name("");
		const xml_attribute attr_name = node_attribute.attribute("name");
		if (attr_name)
			name = attr_name.value();

		TypeDesc type;

		READ_ATTR_I(basetype, unsigned char);      ///< C data type at the heart of our type
		READ_ATTR_I(aggregate, unsigned char);     ///< What kind of AGGREGATE is it?
		READ_ATTR_I(vecsemantics, unsigned char);  ///< Hint: What does the aggregate represent?
		READ_ATTR_I(reserved, unsigned char);      ///< Reserved for future expansion
		READ_ATTR_I(arraylen, int);      ///< Array length, 0 = not array, -1 = unsized

		type.basetype = basetype;
		type.aggregate = aggregate;
		type.vecsemantics = vecsemantics;
		type.reserved = reserved;
		type.arraylen = arraylen;

		const xml_attribute attr_data = node_attribute.attribute("data");
		vector<char> data;
		if (attr_data) {
			read_vector_from_binary_file(state, data, attr_data.value());
		}

		READ_ATTR_I(interp, ParamValue::Interp);
		ParamValue param_value(name, type, data.size() / type.size(), interp, (void*)data.data());
		object->attributes.push_back(param_value);
	}

	if (object->get_geometry() == nullptr)
		return; // TODO

	//scene->objects.push_back(object);
}

/* Scene */

static void xml_read_include(XMLReadState& state, const string& src);

static void xml_read_scene(XMLReadState& state, const xml_node scene_node)
{
	for (const xml_node node : scene_node.children()) {
		if (string_iequals(node.name(), "film")) {
			xml_read_node(state, state.scene->film, node);
		}
		else if (string_iequals(node.name(), "integrator")) {
			xml_read_node(state, state.scene->integrator, node);
		}
		else if (string_iequals(node.name(), "camera")) {
			xml_read_camera(state, node, state.scene->camera);
		}
		else if (string_iequals(node.name(), "dicing_camera")) {
			xml_read_camera(state, node, state.scene->dicing_camera);
		}
		else if (string_iequals(node.name(), "shader")) {
			xml_read_shader(state, node);
		}
		else if (string_iequals(node.name(), "background")) {
			xml_read_background(state, node);
		}
		else if (
			string_iequals(node.name(), "light") ||
			string_iequals(node.name(), "mesh") ||
			string_iequals(node.name(), "hair") ||
			string_iequals(node.name(), "volume") ||
			string_iequals(node.name(), "pointcloud")) {
			xml_read_geom(state, node);
		}
		//else if (string_iequals(node.name(), "light")) {
		//	xml_read_light(state, node);
		//}
		else if (string_iequals(node.name(), "particle_system")) {
			//xml_read_particle_system(state, node);
			ParticleSystem* ps = state.scene->create_node<ParticleSystem>();
			xml_read_node(state, ps, node);
		}
		else if (string_iequals(node.name(), "pass")) {
		    //xml_read_pass(state, node);
			Pass* p = state.scene->create_node<Pass>();
			xml_read_node(state, p, node);
		}
		else if (string_iequals(node.name(), "procedural")) {
		    //xml_read_procedural(state, node);
			//Procedural* p = state.scene->create_node<Procedural>();
			//xml_read_node(state, p, node);
		}
		else if (string_iequals(node.name(), "object")) {
			xml_read_object(state, node);
		}
		//#else
		//    else if (string_iequals(node.name(), "transform")) {
		//      XMLReadState substate = state;
		//
		//      xml_read_transform(node, substate.tfm);
		//      xml_read_scene(substate, node);
		//    }
		//    else if (string_iequals(node.name(), "state")) {
		//      XMLReadState substate = state;
		//
		//      xml_read_state(substate, node);
		//      xml_read_scene(substate, node);
		//    }
		//    else if (string_iequals(node.name(), "object")) {
		//      XMLReadState substate = state;
		//
		//      xml_read_object(substate, node);
		//      xml_read_scene(substate, node);
		//    }
		//#ifdef WITH_ALEMBIC
		//    else if (string_iequals(node.name(), "alembic")) {
		//      xml_read_alembic(state, node);
		//    }
		//#endif
		//#endif
		else if (string_iequals(node.name(), "include")) {
			string src;

			if (xml_read_string(&src, node, "src")) {
				xml_read_include(state, src);
			}
		}
		else {
			fprintf(stderr, "Unknown node \"%s\".\n", node.name());
		}
	}
}

/* Include */

static void xml_read_include(XMLReadState& state, const string& src_xml)
{
	/* open XML document */
	xml_document doc;
	xml_parse_result parse_result;

	string filename_xml = string(src_xml);// +string(".xml");
	string filename_bin = string(src_xml) + string(".bin");

	std::cout << "Loading scene from " << filename_xml << " and " << filename_bin << "\n";

	// Open the file in binary read mode
	state.file = std::make_shared<std::ifstream>(filename_bin, std::ios::binary);
	if (!state.file->is_open()) {
		std::cerr << "Error: Could not open file for reading: " << filename_bin << "\n";
		return;
	}

	//string path = path_join(state.base, src);
	parse_result = doc.load_file(filename_xml.c_str());

	if (parse_result) {
		const xml_node cycles = doc.child("cycles");
		xml_read_scene(state, cycles);
	}
	else {
		fprintf(stderr, "%s read error: %s\n", filename_xml.c_str(), parse_result.description());
		exit(EXIT_FAILURE);
	}

	state.file->close();
}

/* File */

void xml_read_file(Scene* scene, const char* filepath)
{
	XMLReadState state;

	state.scene = scene;

	xml_read_include(state, filepath);
	scene->params.bvh_type = BVH_TYPE_STATIC; //TODO?
}

#if 1
void xml_set_volume_to_attr(Scene* scene, std::string geom_name, std::string attr_name, int type, vector<char>& file_content)
{
	for (Geometry* geom : scene->geometry) {
		if (geom->name == ustring(geom_name)) {
			for (Attribute& attr : geom->attributes.attributes) {

				if (attr.name == attr_name) {
					//openvdb::GridBase::Ptr float_grid;

					openvdb::initialize();

					device_texture* dt = attr.data_voxel().image_memory();
					//dt->device_free();
					//dt->host_free();

					unique_ptr<ImageLoader> loader = nullptr;

					if (type == FTI_OPENVDB) {
						// Convert the vector<uint8_t> back into a stringstream
						std::string str(file_content.begin(), file_content.end());
						std::istringstream stream(str, std::ios_base::binary);

						// Use OpenVDB's Stream to read the grid
						openvdb::io::Stream vdb_stream(stream);
						openvdb::GridPtrVecPtr grids = vdb_stream.getGrids();
						openvdb::GridBase::Ptr float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grids->at(0));

						//loader = new VDBImageLoader(float_grid, attr_name);
						loader = make_unique<VDBImageLoader>(float_grid, attr_name);
						//nanovdb::GridHandle<> nanovdb_handle = VDBImageLoader::convert(float_grid);
						//if (nanovdb_handle) {
						//	dt->alloc(nanovdb_handle.size(), 0);
						//	memcpy(dt->host_pointer, nanovdb_handle.data(), nanovdb_handle.size());
						//	VDBImageLoader::get_texture_info((nanovdb::NanoGrid<float>*)nanovdb_handle.data(), nanovdb_handle.size(), dt->info);
						//}
					}
					else if (type == FTI_PATH) {
						std::string filename(file_content.begin(), file_content.end());
						openvdb::io::File vdbFile(filename);
						vdbFile.open(); // Explicitly open the file
						openvdb::GridBase::Ptr float_grid = vdbFile.readGrid(attr_name); // Pass `name` directly if it is std::string
						vdbFile.close(); // Close the file after reading

						//loader = new VDBImageLoader(float_grid, attr_name);
						loader = make_unique<VDBImageLoader>(float_grid, attr_name);

						//nanovdb::GridHandle<> nanovdb_handle = VDBImageLoader::convert(float_grid);
						//if (nanovdb_handle) {
						//	dt->alloc(nanovdb_handle.size(), 0);
						//	memcpy(dt->host_pointer, nanovdb_handle.data(), nanovdb_handle.size());
						//	VDBImageLoader::get_texture_info((nanovdb::NanoGrid<float>*)nanovdb_handle.data(), nanovdb_handle.size(), dt->info);
						//}
					}
					else if (type == FTI_NANOVDB) {
						//nanovdb::NanoGrid<float>* nanogrid = nullptr;
						//size_t nanogrid_size = 0;
						//vector<char> nanogrid;
						//std::string filename = attr_buffer.value();

						//if (xml_is_digit(filename)) {
						//	//
						//	read_vector_from_binary_file(state, nanogrid);
						//}
						//else {
						//	nanovdb::GridHandle<nanovdb::HostBuffer> grid_handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(filename);
						//	size_t nanogrid_size = grid_handle.size();
						//	nanogrid.resize(nanogrid_size);
						//	memcpy(nanogrid.data(), grid_handle.data(), nanogrid_size);
						//}

						//loader = new NanoVDBImageLoader(file_content);
						loader = make_unique<NanoVDBImageLoader>(file_content);

						//if (dt && file_content.size() > 0) {
						//	dt->alloc(file_content.size(), 0);
						//	memcpy(dt->host_pointer, file_content.data(), file_content.size());
						//	VDBImageLoader::get_texture_info((nanovdb::NanoGrid<float>*)file_content.data(), file_content.size(), dt->info);
						//}

						//return;
					}

					//else {
					//	return;
					//}

					//ImageLoader* loader = new VDBImageLoader(float_grid, attr_name);
					//ImageParams params;
					//int slot = attr.data_voxel().svm_slot(0);
					//attr.data_voxel() = scene->image_manager->add_image(loader, params, false); // TODO remove???

					//unique_ptr<ImageLoader> loader = make_unique<VDBImageLoader>(grid, name.string());
					const ImageParams params;
					attr.data_voxel() = scene->image_manager->add_image(std::move(loader), params);

					//geom->tag_update(scene, true);
					geom->tag_modified();
					geom->tag_update(scene, true);

					//break;

					//dt->copy_to_device();

					return;
				}
			}
		}
	}
}
#else
void xml_set_volume_to_attr(Scene* scene, std::string geom_name, std::string attr_name, int type, vector<char>& file_content)
{
	for(Geometry * geom: scene->geometry) {
		if (geom->name == ustring(geom_name)) {
			for(Attribute & attr: geom->attributes.attributes) {

				if (attr.name == attr_name) {				
					//openvdb::GridBase::Ptr float_grid;

					openvdb::initialize();

					device_texture* dt = attr.data_voxel().image_memory();
					//dt->device_free();
					//dt->host_free();

					//ImageLoader* loader = nullptr;		

					if (type == FTI_OPENVDB) {
						// Convert the vector<uint8_t> back into a stringstream
						std::string str(file_content.begin(), file_content.end());
						std::istringstream stream(str, std::ios_base::binary);

						// Use OpenVDB's Stream to read the grid
						openvdb::io::Stream vdb_stream(stream);
						openvdb::GridPtrVecPtr grids = vdb_stream.getGrids();
						openvdb::GridBase::Ptr float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grids->at(0));

						//loader = new VDBImageLoader(float_grid, attr_name);
						nanovdb::GridHandle<> nanovdb_handle = VDBImageLoader::convert(float_grid);
						if (nanovdb_handle) {
							dt->alloc(nanovdb_handle.size(), 0);
							memcpy(dt->host_pointer, nanovdb_handle.data(), nanovdb_handle.size());
							VDBImageLoader::get_texture_info((nanovdb::NanoGrid<float>*)nanovdb_handle.data(), nanovdb_handle.size(), dt->info);
						}
					}
					else if (type == FTI_PATH) {
						std::string filename(file_content.begin(), file_content.end());
						openvdb::io::File vdbFile(filename);
						vdbFile.open(); // Explicitly open the file
						openvdb::GridBase::Ptr float_grid = vdbFile.readGrid(attr_name); // Pass `name` directly if it is std::string
						vdbFile.close(); // Close the file after reading

						//loader = new VDBImageLoader(float_grid, attr_name);

						nanovdb::GridHandle<> nanovdb_handle = VDBImageLoader::convert(float_grid);
						if (nanovdb_handle) {
							dt->alloc(nanovdb_handle.size(), 0);
							memcpy(dt->host_pointer, nanovdb_handle.data(), nanovdb_handle.size());
							VDBImageLoader::get_texture_info((nanovdb::NanoGrid<float>*)nanovdb_handle.data(), nanovdb_handle.size(), dt->info);
						}
					}
					else if (type == FTI_NANOVDB) {
						//nanovdb::NanoGrid<float>* nanogrid = nullptr;
						//size_t nanogrid_size = 0;
						//vector<char> nanogrid;
						//std::string filename = attr_buffer.value();

						//if (xml_is_digit(filename)) {
						//	//
						//	read_vector_from_binary_file(state, nanogrid);
						//}
						//else {
						//	nanovdb::GridHandle<nanovdb::HostBuffer> grid_handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(filename);
						//	size_t nanogrid_size = grid_handle.size();
						//	nanogrid.resize(nanogrid_size);
						//	memcpy(nanogrid.data(), grid_handle.data(), nanogrid_size);
						//}
						
						//loader = new NanoVDBImageLoader(file_content);					

						if (dt && file_content.size() > 0) {
							dt->alloc(file_content.size(), 0);
							memcpy(dt->host_pointer, file_content.data(), file_content.size());
							VDBImageLoader::get_texture_info((nanovdb::NanoGrid<float>*)file_content.data(), file_content.size(), dt->info);
						}

						//return;
					}

					//else {
					//	return;
					//}

					//ImageLoader* loader = new VDBImageLoader(float_grid, attr_name);
					//ImageParams params;
					////int slot = attr.data_voxel().svm_slot(0);
					//attr.data_voxel() = scene->image_manager->add_image(loader, params, false); // TODO remove???
					//geom->tag_update(scene, true);

					//break;

					dt->copy_to_device();

					geom->tag_modified();
					geom->tag_update(scene, true);

					return;
				}
			}
		}
	}
}
#endif

void xml_set_material_to_node(Scene* scene, const char* file_content)
{
	if (file_content == nullptr || strlen(file_content) == 0)
		return;

	xml_document doc;
	xml_parse_result parse_result;
	parse_result = doc.load_string(file_content);
	if (parse_result) {
		xml_node xnode_shader = doc.child("shader");
		XMLReader graph_reader;
		Shader bshader_temp;
		xml_read_node(graph_reader, &bshader_temp, xnode_shader);

		for(Shader * shader: scene->shaders) {

			if (shader->name == bshader_temp.name) {
				
				//graph_reader.node_map[ustring("output")] = shader->graph->output();
				//graph_reader.file = state.file;

				for (xml_node xnode_node : xnode_shader.children()) {
					
					for(ShaderNode * snode: shader->graph->nodes) {

						const xml_attribute name_attr = xnode_node.attribute("name");
						if (name_attr) {
							ustring xname = ustring(name_attr.value());

							if (xname == snode->name) {

								xml_read_node(graph_reader, snode, xnode_node);

								shader->tag_update(scene);

								return;
							}
						}
					}
				}
			}
		}
	}
}

void xml_set_material_to_shader(Scene* scene, const char* file_content)
{
	if (file_content == nullptr || strlen(file_content) == 0)
		return;

	xml_document doc;
	xml_parse_result parse_result;
	parse_result = doc.load_string(file_content);
	if (parse_result) {
		xml_node xnode_shader = doc.child("shader");
		XMLReader graph_reader;
		Shader bshader_temp;
		xml_read_node(graph_reader, &bshader_temp, xnode_shader);

		for(Shader * shader: scene->shaders) {

			if (shader->name == bshader_temp.name) {

				//graph_reader.node_map[ustring("output")] = shader->graph->output();
				//graph_reader.file = state.file;

				XMLReadState state;
				state.scene = scene;

				xml_read_shader_graph(state, shader, xnode_shader);
			}
		}
	}
}

void xml_set_material_to_shader2(Scene* scene, Shader* shader, const char* file_content)
{
	if (file_content == nullptr || strlen(file_content) == 0)
		return;

	xml_document doc;
	//xml_parse_result parse_result;
	//parse_result = doc.load_string(file_content);
	std::ifstream in(file_content, std::ios::binary);
	std::string file_data((std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());

	xml_parse_result parse_result = doc.load_buffer(file_data.data(), file_data.size());
	if (parse_result) {
		xml_node xnode_shader = doc.child("shader");
		XMLReader graph_reader;
		xml_read_node(graph_reader, shader, xnode_shader);

		XMLReadState state;
		state.scene = scene;

		xml_read_shader_graph(state, shader, xnode_shader);
	}
}

CCL_NAMESPACE_END
