/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene_write_xml.h"

#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <sstream>

#include "scene/node_write_xml.h"

//#include "scene/alembic.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/osl.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"
#include "scene/image_vdb.h"

#include "subd/patch.h"
#include "subd/split.h"

//#include "util/foreach.h"
#include "util/path.h"
#include "util/projection.h"
#include "util/transform.h"
#include "util/xml.h"
#include "util/string.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/filesystem.h>

#ifdef WITH_OPENVDB
#	include <openvdb/io/Stream.h>
#endif

#include "graph/node_xml_util.h"

#define ADD_ATTR(name) \
			xml_attribute attr_##name = node_attribute.append_attribute(#name); \
			attr_##name = attr.name;

#define ADD_ATTR_ENUM(name) \
			xml_attribute attr_##name = node_attribute.append_attribute(#name); \
			attr_##name = enum_to_str(attr.name);

#define ADD_ATTR_STR(name) \
			xml_attribute attr_##name = node_attribute.append_attribute(#name); \
			attr_##name = attr.name.c_str();

#define ADD_ATTR_DTYPE(type, name) \
			xml_attribute attr_##name = node_attribute.append_attribute(#name); \
			attr_##name = type.name;

#define ADD_ATTR_DTYPE_DESC(type, name) \
			xml_attribute attr_##name = node_attribute.append_attribute(#name); \
			attr_##name = typedesc_to_cstr(type);

#define ADD_ATTR_TYPE_DESC(name) \
			xml_attribute attr_##name = node_attribute.append_attribute(#name); \
			attr_##name = typedesc_to_cstr(attr.name);

CCL_NAMESPACE_BEGIN

/* XML writing state */

struct XMLWriteState : public XMLWriter {
	Scene* scene;      /* Scene pointer. */	

	XMLWriteState() :
		scene(NULL)
	{
	}
};

template<typename T>
string write_vector_to_binary_file(XMLWriter& writer, const vector<T>& data)
{
	std::stringstream ss;
	ss << writer.offset;

	// Write the size of the vector first
	size_t data_size = data.size();
	writer.file.write(reinterpret_cast<const char*>(&data_size), sizeof(size_t));
	writer.offset += sizeof(size_t);

	// Write the actual data to the file
	writer.file.write(reinterpret_cast<const char*>(data.data()), data_size * sizeof(T));
	writer.offset += data_size * sizeof(T);

	return ss.str();
}

void save_image_to_memory(device_texture* dt, vector<char>& output_buffer) {
#if 0
	TypeDesc image_type;
	switch (dt->data_type) {
	case ImageDataType::IMAGE_DATA_TYPE_FLOAT4:
	case ImageDataType::IMAGE_DATA_TYPE_FLOAT:
		image_type = TypeDesc::FLOAT;
		break;
	case ImageDataType::IMAGE_DATA_TYPE_BYTE4:
	case ImageDataType::IMAGE_DATA_TYPE_BYTE:
		image_type = TypeDesc::UCHAR;
		break;
	case ImageDataType::IMAGE_DATA_TYPE_HALF4:
	case ImageDataType::IMAGE_DATA_TYPE_HALF:
		image_type = TypeDesc::HALF;
		break;
	case ImageDataType::IMAGE_DATA_TYPE_USHORT4:
	case ImageDataType::IMAGE_DATA_TYPE_USHORT:
		image_type = TypeDesc::USHORT;
		break;
	default:
		std::cerr << "Wrong image type" << std::endl;
		return;
	}

	int width = dt->data_width;
	int height = dt->data_height;
	int channels = dt->data_elements;
	ImageSpec spec(width, height, channels, image_type);

	// Use OpenImageIO's ImageBuf for flipping
	ImageBuf img_buf(spec, dt->host_pointer);
	ImageBuf flipped_buf;
	if (!ImageBufAlgo::flip(flipped_buf, img_buf)) {
		std::cerr << "Error flipping the image" << std::endl;
		return;
	}

	// Create an in-memory image writer using IOVecOutput
	std::unique_ptr<ImageOutput> image_output = ImageOutput::create("image.exr");
	if (!image_output) {
		std::cerr << "Failed to create image output for memory buffer" << std::endl;
		return;
	}

	std::vector<unsigned char> io_buffer;
	Filesystem::IOVecOutput io_vec_output(io_buffer);
	//void* ptr = &io_vec_output;
	//spec.attribute("oiio:ioproxy", TypeDesc::PTR, &ptr);
	image_output->set_ioproxy(&io_vec_output);
	image_output->open("image.exr", spec);

	if (!image_output->write_image(image_type, flipped_buf.localpixels())) { //flipped_buf
		std::cerr << "Error writing image" << std::endl;
		return;
	}
	image_output->close();
#endif

#if 0
	struct TempMetaData {
		int width;
		int height;
		int channels;
		TypeDesc image_type;
	};
	TempMetaData temp_meta_data;
	temp_meta_data.width = width;
	temp_meta_data.height = height;
	temp_meta_data.channels = channels;
	temp_meta_data.image_type = image_type;	

	// Get the buffer as a vector<char>
	output_buffer.resize(sizeof(TempMetaData));
	memcpy(output_buffer.data(), &temp_meta_data, sizeof(TempMetaData));

	output_buffer.insert(output_buffer.end(), io_buffer.begin(), io_buffer.end());
	//output_buffer.assign(io_buffer.begin(), io_buffer.end());
#endif

	output_buffer.resize(dt->memory_size());
	memcpy(output_buffer.data(), dt->host_pointer, dt->memory_size());
}

/* Attribute Reading */

void scene_write_xml_int(int value, xml_node node, const char* name)
{
	xml_attribute attr = node.append_attribute(name);

	attr.set_value(value);
}

void scene_write_xml_int_array(vector<int>& value, xml_node node, const char* name)
{
	xml_attribute attr = node.append_attribute(name);

	std::ostringstream oss;

	for (size_t i = 0; i < value.size(); ++i) {
		if (i != 0 && i != value.size() - 1) {
			oss << ";"; // Add separator before each number except the first and the last
		}
		oss << value[i];
	}

	attr.set_value(oss.str().c_str());
}

void scene_write_xml_float(float value, xml_node node, const char* name)
{
	xml_attribute attr = node.append_attribute(name);

	attr.set_value(value);
}

void scene_write_xml_float_array(vector<float>& value, xml_node node, const char* name)
{
	xml_attribute attr = node.append_attribute(name);

	std::ostringstream oss;

	for (size_t i = 0; i < value.size(); ++i) {
		if (i != 0 && i != value.size() - 1) {
			oss << ";"; // Add separator before each number except the first and the last
		}
		oss << value[i];
	}

	attr.set_value(oss.str().c_str());
}

void scene_write_xml_float3(float3 value, xml_node node, const char* name)
{
	vector<float> array;
	array.push_back(value[0]);
	array.push_back(value[1]);
	array.push_back(value[2]);

	scene_write_xml_float_array(array, node, name);
}

void scene_write_xml_float3_array(vector<float3>& value, xml_node node, const char* name)
{
	vector<float> array;

	for(float3 v: value) {
		array.push_back(v[0]);
		array.push_back(v[1]);
		array.push_back(v[2]);
	}

	scene_write_xml_float_array(array, node, name);
}

void scene_write_xml_float4(float4 value, xml_node node, const char* name)
{
	vector<float> array;
	array.push_back(value[0]);
	array.push_back(value[1]);
	array.push_back(value[2]);
	array.push_back(value[3]);

	scene_write_xml_float_array(array, node, name);
}

void scene_write_xml_string(string str, xml_node node, const char* name)
{
	xml_attribute attr = node.append_attribute(name);
	attr.set_value(str.c_str());
}

//bool scene_write_xml_equal_string(xml_node node, const char* name, const char* value)
//{
//	xml_attribute attr = node.append_attribute(name);
//
//	if (attr) {
//		return string_iequals(attr.value(), value);
//	}
//
//	return false;
//}

/* Camera */

//void scene_write_xml_camera(Scene* scene, xml_node node)
//{
//	Camera* cam = scene->camera;
//
//	int width = cam->get_full_width(), height = cam->get_full_height();
//	scene_write_xml_int(width, node, "width");
//	scene_write_xml_int(height, node, "height");
//
//	//cam->set_full_width(width);
//	//cam->set_full_height(height);
//
//	xml_write_node(cam, node);
//
//	//cam->set_matrix(state.tfm);
//
//	//cam->need_flags_update = true;
//	//cam->update(state.scene);
//}

/* Alembic */

//#ifdef WITH_ALEMBIC
//void scene_write_xml_alembic(XMLReadState& state, xml_node graph_node)
//{
//    AlembicProcedural* proc = state.scene->create_node<AlembicProcedural>();
//    xml_write_node(state, proc, graph_node);
//
//    for (xml_node node = graph_node.first_child(); node; node = node.next_sibling()) {
//        if (string_iequals(node.name(), "object")) {
//            string path;
//            if (scene_write_xml_string(&path, node, "path")) {
//                ustring object_path(path, 0);
//                AlembicObject* object = static_cast<AlembicObject*>(
//                    proc->get_or_create_object(object_path));
//
//                array<Node*> used_shaders = object->get_used_shaders();
//                used_shaders.push_back_slow(state.shader);
//                object->set_used_shaders(used_shaders);
//            }
//        }
//    }
//}
//#endif

/* Shader */

void scene_write_xml_shader_graph(XMLWriteState& state, Shader* shader, xml_node xml_root)
{
	//xml_node graph_node = xml_root;// .append_child(shader->type->name.c_str());
	//xml_write_node(shader, graph_node);

	for(ShaderNode * node: shader->graph->nodes) {
		if (node->name == "output")
			continue; // skip

		//xml_node xml_node = xml_root.append_child(node->type->name.c_str());
		xml_node xnode = xml_write_node(state, node, xml_root);
		if (node->type->name == "image_texture" || node->type->name == "environment_texture") {

			bool socket_found = false;

			xml_node xml_node_socket_found;
			xml_attribute attr_name_found;
			xml_attribute attr_name_ui_found;
			xml_attribute attr_value_found;

			for (xml_node xml_node_socket : xnode.children("socket")) {

				xml_attribute attr_name = xml_node_socket.attribute("name");
				xml_attribute attr_name_ui = xml_node_socket.attribute("ui_name");
				xml_attribute attr_value = xml_node_socket.attribute("value");

				//ustring xml_socket_name(attr_name.value());

				if (attr_name && ustring(attr_name.value()) == "filename" || 
					attr_name_ui && ustring(attr_name_ui.value()) == "Filename") {
					//std::string str_filename(attr.value());
					//if (!str_filename.empty()) {

					xml_node_socket_found = xml_node_socket;
					attr_name_found = attr_name;
					attr_name_ui_found = attr_name_ui;
					attr_value_found = attr_value;

					socket_found = true;
					break;
					//}
				}
			}

			//if (socket_found)
			//	continue;

			//xml_attribute attr_filename = xnode.attribute("filename");
			//if (attr_filename) {
			//	std::string str_filename(attr_filename.value());
			//	if (!str_filename.empty())
			//		continue;
			//}
			//else {
			//	attr_filename = xnode.append_attribute("filename");
			//}

			//xml_node xml_node_socket_found;
			//xml_attribute attr_name_found;
			//xml_attribute attr_name_ui_found;
			//xml_attribute attr_value_found;

			if (!socket_found) {
				xml_node_socket_found = xnode.append_child("socket");
				attr_name_found = xml_node_socket_found.append_attribute("name");
				attr_name_found = "filename";

				attr_name_ui_found = xml_node_socket_found.append_attribute("ui_name");
				attr_name_ui_found = "Filename";

				attr_value_found = xml_node_socket_found.append_attribute("value");
			}

			ImageSlotTextureNode* img = (ImageSlotTextureNode*)node;
			//ImageMetaData metadata = img->handle.metadata();

			device_texture* dt = img->handle.image_memory();
			if (dt && dt->host_pointer) {
#if 0
				std::stringstream ss;

				ss << xml_pointer_to_name(dt->host_pointer) << ".exr";
				std::string str = ss.str();
				std::replace(str.begin(), str.end(), ' ', '_');
				string filename(path_join(state.base, str));

				unique_ptr<ImageOutput> image_output(ImageOutput::create(filename));
				if (!image_output) {
					fprintf(stderr, "Failed to create image output for %s\n", filename.c_str());
					return;
				}

				TypeDesc image_type;
				switch (metadata.type) {
				case ImageDataType::IMAGE_DATA_TYPE_FLOAT4:
				case ImageDataType::IMAGE_DATA_TYPE_FLOAT:
					image_type = TypeDesc::FLOAT;
					break;
				case ImageDataType::IMAGE_DATA_TYPE_BYTE4:
				case ImageDataType::IMAGE_DATA_TYPE_BYTE:
					image_type = TypeDesc::UCHAR;
					break;
				case ImageDataType::IMAGE_DATA_TYPE_HALF4:
				case ImageDataType::IMAGE_DATA_TYPE_HALF:
					image_type = TypeDesc::HALF;
					break;
				case ImageDataType::IMAGE_DATA_TYPE_USHORT4:
				case ImageDataType::IMAGE_DATA_TYPE_USHORT:
					image_type = TypeDesc::USHORT;
					break;
					//case ImageDataType::IMAGE_DATA_TYPE_NANOVDB_FLOAT:
					//case ImageDataType::IMAGE_DATA_TYPE_NANOVDB_FLOAT3:
					//case ImageDataType::IMAGE_DATA_TYPE_NANOVDB_FPN:
					//case ImageDataType::IMAGE_DATA_TYPE_NANOVDB_FP16:
					//case ImageDataType::IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT:
				default:
					fprintf(stderr, "Wrong image type");
					return;
				}

				// Setup image specification
				int width = metadata.width;
				int height = metadata.height;
				ImageSpec spec(width, height, metadata.channels, image_type);

				// Use OpenImageIO's ImageBuf for flipping
				ImageBuf img_buf(spec, dt->host_pointer);
				ImageBuf flipped_buf;

				if (!ImageBufAlgo::flip(flipped_buf, img_buf)) {
					fprintf(stderr, "Error flipping the image");
					return;
				}

				if (!image_output->open(filename, spec)) {
					fprintf(stderr, "Failed to open image file for %s\n", filename.c_str());
					return;
				}

				if (!image_output->write_image(image_type, flipped_buf.localpixels())) {
					fprintf(stderr, "Error writing image");
					return;
				}

				// Close the file
				if (!image_output->close()) {
					fprintf(stderr, "Error closing file");
					return;
				}
				attr_filename = filename.c_str();
#endif
				vector<char> output_buffer;
				save_image_to_memory(dt, output_buffer);
				std::string soffset = write_vector_to_binary_file(state, output_buffer);
				attr_value_found = soffset.c_str();

				// MetaData
				xml_node node_attribute = xnode;
				ImageMetaData attr = img->handle.metadata();

				bool is_rgba = (attr.type == IMAGE_DATA_TYPE_FLOAT4 ||
					attr.type == IMAGE_DATA_TYPE_HALF4 ||
					attr.type == IMAGE_DATA_TYPE_BYTE4 ||
					attr.type == IMAGE_DATA_TYPE_USHORT4);

				if (is_rgba)
					attr.channels = 4; //ImageManager::file_load_image

				//int channels;			
				ADD_ATTR(channels);

				//size_t width, height, depth;
				ADD_ATTR(width);
				ADD_ATTR(height);
				//ADD_ATTR(depth);
				ADD_ATTR(type);

				///* Optional color space, defaults to raw. */
				//ustring colorspace;
				ADD_ATTR_STR(colorspace);
				//string colorspace_file_hint;
				ADD_ATTR_STR(colorspace_file_hint);
				//const char* colorspace_file_format;
				ADD_ATTR(colorspace_file_format);

				///* Optional transform for 3D images. */
				//bool use_transform_3d;
				//Transform transform_3d;

				///* Automatically set. */
				//bool compress_as_srgb;
			}
		}
	}

	for(ShaderNode * node: shader->graph->nodes) {
		for(ShaderOutput * output: node->outputs) {
			for(ShaderInput * input: output->links) {
				xml_node connect_node = xml_root.append_child("connect");

				xml_attribute attr_from_node = connect_node.append_attribute("from_node");
				attr_from_node = xml_pointer_to_name(output->parent).c_str();

				xml_attribute attr_from_socket = connect_node.append_attribute("from_socket");
				attr_from_socket = output->socket_type.name.c_str();

				xml_attribute attr_from_socket_ui = connect_node.append_attribute("from_socket_ui");
				attr_from_socket_ui = output->socket_type.ui_name.c_str();

				xml_attribute attr_to_node = connect_node.append_attribute("to_node");
				string to_name = xml_pointer_to_name(input->parent);
				if (dynamic_cast<OutputNode*>(input->parent))
					to_name = input->parent->name.string(); // output
				attr_to_node = to_name.c_str();
					
				xml_attribute attr_to_socket = connect_node.append_attribute("to_socket");
				attr_to_socket = input->socket_type.name.c_str();

				xml_attribute attr_to_socket_ui = connect_node.append_attribute("to_socket_ui");
				attr_to_socket_ui = input->socket_type.ui_name.c_str();
			}
		}
	}
	//
	//
	//	xml_write_node(state, shader, graph_node);
	//
	//	ShaderGraph* graph = new ShaderGraph();
	//
	//	/* local state, shader nodes can't link to nodes outside the shader graph */
	//	XMLReader graph_reader;
	//	graph_reader.node_map[ustring("output")] = graph->output();
	//
	//	for (xml_node node = graph_node.first_child(); node; node = node.next_sibling()) {
	//		ustring node_name(node.name());
	//
	//		if (node_name == "connect") {
	//			/* connect nodes */
	//			vector<string> from_tokens, to_tokens;
	//
	//			string_split(from_tokens, node.append_attribute("from").value());
	//			string_split(to_tokens, node.append_attribute("to").value());
	//
	//			if (from_tokens.size() == 2 && to_tokens.size() == 2) {
	//				ustring from_node_name(from_tokens[0]);
	//				ustring from_socket_name(from_tokens[1]);
	//				ustring to_node_name(to_tokens[0]);
	//				ustring to_socket_name(to_tokens[1]);
	//
	//				/* find nodes and sockets */
	//				ShaderOutput* output = NULL;
	//				ShaderInput* input = NULL;
	//
	//				if (graph_reader.node_map.find(from_node_name) != graph_reader.node_map.end()) {
	//					ShaderNode* fromnode = (ShaderNode*)graph_reader.node_map[from_node_name];
	//
	//					foreach(ShaderOutput * out, fromnode->outputs)
	//						if (string_iequals(out->socket_type.name.string(), from_socket_name.string())) {
	//							output = out;
	//						}
	//
	//					if (!output) {
	//						fprintf(stderr,
	//							"Unknown output socket name \"%s\" on \"%s\".\n",
	//							from_node_name.c_str(),
	//							from_socket_name.c_str());
	//					}
	//				}
	//				else {
	//					fprintf(stderr, "Unknown shader node name \"%s\".\n", from_node_name.c_str());
	//				}
	//
	//				if (graph_reader.node_map.find(to_node_name) != graph_reader.node_map.end()) {
	//					ShaderNode* tonode = (ShaderNode*)graph_reader.node_map[to_node_name];
	//
	//					foreach(ShaderInput * in, tonode->inputs)
	//						if (string_iequals(in->socket_type.name.string(), to_socket_name.string())) {
	//							input = in;
	//						}
	//
	//					if (!input) {
	//						fprintf(stderr,
	//							"Unknown input socket name \"%s\" on \"%s\".\n",
	//							to_socket_name.c_str(),
	//							to_node_name.c_str());
	//					}
	//				}
	//				else {
	//					fprintf(stderr, "Unknown shader node name \"%s\".\n", to_node_name.c_str());
	//				}
	//
	//				/* connect */
	//				if (output && input) {
	//					graph->connect(output, input);
	//				}
	//			}
	//			else {
	//				fprintf(stderr, "Invalid from or to value for connect node.\n");
	//			}
	//
	//			continue;
	//		}
	//
	//		ShaderNode* snode = NULL;
	//
	//#ifdef WITH_OSL
	//		if (node_name == "osl_shader") {
	//			ShaderManager* manager = state.scene->shader_manager;
	//
	//			if (manager->use_osl()) {
	//				std::string filepath;
	//
	//				if (scene_write_xml_string(&filepath, node, "src")) {
	//					if (path_is_relative(filepath)) {
	//						filepath = path_join(state.base, filepath);
	//					}
	//
	//					snode = OSLShaderManager::osl_node(graph, manager, filepath, "");
	//
	//					if (!snode) {
	//						fprintf(stderr, "Failed to create OSL node from \"%s\".\n", filepath.c_str());
	//						continue;
	//					}
	//				}
	//				else {
	//					fprintf(stderr, "OSL node missing \"src\" attribute.\n");
	//					continue;
	//				}
	//			}
	//			else {
	//				fprintf(stderr, "OSL node without using --shadingsys osl.\n");
	//				continue;
	//			}
	//		}
	//		else
	//#endif
	//		{
	//			/* exception for name collision */
	//			if (node_name == "background") {
	//				node_name = "background_shader";
	//			}
	//
	//			const NodeType* node_type = NodeType::find(node_name);
	//
	//			if (!node_type) {
	//				fprintf(stderr, "Unknown shader node \"%s\".\n", node.name());
	//				continue;
	//			}
	//			else if (node_type->type != NodeType::SHADER) {
	//				fprintf(stderr, "Node type \"%s\" is not a shader node.\n", node_type->name.c_str());
	//				continue;
	//			}
	//			else if (node_type->create == NULL) {
	//				fprintf(stderr, "Can't create abstract node type \"%s\".\n", node_type->name.c_str());
	//				continue;
	//			}
	//
	//			snode = (ShaderNode*)node_type->create(node_type);
	//			snode->set_owner(graph);
	//		}
	//
	//		xml_write_node(graph_reader, snode, node);
	//
	//		if (node_name == "image_texture") {
	//			ImageTextureNode* img = (ImageTextureNode*)snode;
	//			ustring filename(path_join(state.base, img->get_filename().string()));
	//			img->set_filename(filename);
	//		}
	//		else if (node_name == "environment_texture") {
	//			EnvironmentTextureNode* env = (EnvironmentTextureNode*)snode;
	//			ustring filename(path_join(state.base, env->get_filename().string()));
	//			env->set_filename(filename);
	//		}
	//
	//		if (snode) {
	//			/* add to graph */
	//			graph->add(snode);
	//		}
	//	}
	//
	//	shader->set_graph(graph);
	//	shader->tag_update(state.scene);
}

//void scene_write_xml_shader(XMLWriteState& state, xml_node node)
//{
//	//Shader* shader = new Shader();
//	for(Shader * shader: state.scene->shaders) {
//
//		//if (shader->name == "default_background" ||
//		//	shader->name == "default_empty" ||
//		//	shader->name == "default_light" ||
//		//	shader->name == "default_surface" ||
//		//	shader->name == "default_volume")
//		//	continue;
//
//		//xml_node xml_shader = node.append_child(shader->type->name.c_str());
//		xml_node xml_shader = xml_write_node(state, shader, node);
//
//		//if (shader->name == "default_background" ||
//		//	shader->name == "default_empty" ||
//		//	shader->name == "default_light" ||
//		//	shader->name == "default_surface" ||
//		//	shader->name == "default_volume") {
//
//		//	xml_attribute name_attr = xml_shader.attribute("name");
//		//	name_attr = shader->name.c_str();
//		//}
//
//		scene_write_xml_shader_graph(state, shader, xml_shader);
//	}
//	//state.scene->shaders.push_back(shader);
//}

/* Background */

//void scene_write_xml_background(Scene* scene, xml_node node)
//{
//	/* Background Settings */
//	xml_write_node(scene->background, node);
//
//	/* Background Shader */
//	Shader* shader = scene->default_background;
//	xml_node xml_shader = node.append_child(shader->type->name.c_str());
//	xml_write_node(shader, xml_shader);
//
//	scene_write_xml_shader_graph(shader, node);
//}

/* Mesh */

//Mesh* xml_add_mesh(Scene* scene, const Transform& tfm, Object* object)
//{
//    if (object && object->get_geometry()->is_mesh()) {
//        /* Use existing object and mesh */
//        object->set_tfm(tfm);
//        Geometry* geometry = object->get_geometry();
//        return static_cast<Mesh*>(geometry);
//    }
//    else {
//        /* Create mesh */
//        Mesh* mesh = new Mesh();
//        scene->geometry.push_back(mesh);
//
//        /* Create object. */
//        Object* object = new Object();
//        object->set_geometry(mesh);
//        object->set_tfm(tfm);
//        scene->objects.push_back(object);
//
//        return mesh;
//    }
//}
//

void scene_write_xml_geom(XMLWriteState& state, xml_node node)
{
	for(Geometry * geom: state.scene->geometry) {
		//xml_node xml_node = node.append_child(mesh->type->name.c_str());
		xml_node xml_node_geom = xml_write_node(state, geom, node);

		//Type geometry_type;
		xml_attribute attr_gt = xml_node_geom.append_attribute("geometry_type");
		attr_gt = geom->geometry_type;

		if (geom->attributes.attributes.size() > 0) {
			//AttributeSet attributes;
			//xml_node node_attributes = xml_node_geom.append_child("attributes");
			for(Attribute & attr: geom->attributes.attributes) {
				xml_node node_attribute = xml_node_geom.append_child("attribute");

				//ustring name;
				//ADD_ATTR_STR(name);
				xml_attribute attr_name = node_attribute.append_attribute("name");
				attr_name = attr.name.string().c_str();//(xml_pointer_to_name(&attr) + "_" + attr.name.string()).c_str();

				//AttributeStandard std;
				ADD_ATTR(std);				

				//TypeDesc type;
				//ADD_ATTR(type);
				ADD_ATTR_DTYPE(attr.type, basetype);      ///< C data type at the heart of our type
				ADD_ATTR_DTYPE(attr.type, aggregate);     ///< What kind of AGGREGATE is it?
				ADD_ATTR_DTYPE(attr.type, vecsemantics);  ///< Hint: What does the aggregate represent?
				ADD_ATTR_DTYPE(attr.type, reserved);      ///< Reserved for future expansion
				ADD_ATTR_DTYPE(attr.type, arraylen);      ///< Array length, 0 = not array, -1 = unsized

				//AttributeElement element;
				ADD_ATTR(element);

				//uint flags;
				ADD_ATTR(flags);

				//vector<char> buffer;
				xml_attribute attr_buffer = node_attribute.append_attribute("buffer");

				std::string str_buffer = xml_pointer_to_name(&attr) + "_" + attr.name.string();

				std::stringstream ss;								

				if (attr.element == AttributeElement::ATTR_ELEMENT_VOXEL) {
					ImageHandle handle = attr.data_voxel();
					VDBImageLoader *vdb_loader = handle.vdb_loader();

					//vector<char> buffer;
					//// Convert the stringstream to a vector<char>			
					//std::ostringstream stream(std::ios_base::binary);
					//openvdb::io::Stream(stream).write({ vdb_loader->get_grid() });
					//stream.flush();
					//const std::string& str = stream.str();
					//buffer.assign(str.begin(), str.end());

					//ss << write_vector_to_binary_file(state, geom, string("buffer"), buffer);

					//std::stringstream ss;
					//ss << geom->type->name << "_" << geom->name << "_" << attr.name.string() + string("_") + string("buffer") << ".vdb";					
					//ss << attr_name.value() << ".vdb";					
					//std::replace(str.begin(), str.end(), ' ', '_');
					//ss << str_buffer << ".vdb";
					//string filename(path_join(state.base, ss.str()));
					//openvdb::io::File(filename).write({ vdb_loader->get_grid() });
					
#ifdef WITH_OPENVDB  
					// Convert the stringstream to a vector<char>
					vector<char> file_content;
					std::ostringstream stream(std::ios_base::binary);
					openvdb::io::Stream(stream).write({ vdb_loader->get_grid() });
					stream.flush();
					const std::string& str = stream.str();
					file_content.assign(str.begin(), str.end());
					ss << write_vector_to_binary_file(state, file_content);

					xml_attribute attr_volume_type = node_attribute.append_attribute("volume_type");
					attr_volume_type = "openvdb";
#endif				
				}
				else {
					ss << write_vector_to_binary_file(state, attr.buffer);
				}

				attr_buffer = ss.str().c_str();
			}
		}

		//BoundBox bounds;
		//Transform transform_normal;
	}

//    /* add mesh */
//    Mesh* mesh = xml_add_mesh(state.scene, state.tfm, state.object);
//    array<Node*> used_shaders = mesh->get_used_shaders();
//    used_shaders.push_back_slow(state.shader);
//    mesh->set_used_shaders(used_shaders);
//
//    /* read state */
//    int shader = 0;
//    bool smooth = state.smooth;
//
//    /* read vertices and polygons */
//    vector<float3> P;
//    vector<float3> VN; /* Vertex normals */
//    vector<float> UV;
//    vector<float> T;  /* UV tangents */
//    vector<float> TS; /* UV tangent signs */
//    vector<int> verts, nverts;
//
//    scene_write_xml_float3_array(P, node, "P");
//    scene_write_xml_int_array(verts, node, "verts");
//    scene_write_xml_int_array(nverts, node, "nverts");
//
//    if (scene_write_xml_equal_string(node, "subdivision", "catmull-clark")) {
//        mesh->set_subdivision_type(Mesh::SUBDIVISION_CATMULL_CLARK);
//    }
//    else if (scene_write_xml_equal_string(node, "subdivision", "linear")) {
//        mesh->set_subdivision_type(Mesh::SUBDIVISION_LINEAR);
//    }
//
//    array<float3> P_array;
//    P_array = P;
//
//    if (mesh->get_subdivision_type() == Mesh::SUBDIVISION_NONE) {
//        /* create vertices */
//
//        mesh->set_verts(P_array);
//
//        size_t num_triangles = 0;
//        for (size_t i = 0; i < nverts.size(); i++) {
//            num_triangles += nverts[i] - 2;
//        }
//        mesh->reserve_mesh(mesh->get_verts().size(), num_triangles);
//
//        /* create triangles */
//        int index_offset = 0;
//
//        for (size_t i = 0; i < nverts.size(); i++) {
//            for (int j = 0; j < nverts[i] - 2; j++) {
//                int v0 = verts[index_offset];
//                int v1 = verts[index_offset + j + 1];
//                int v2 = verts[index_offset + j + 2];
//
//                assert(v0 < (int)P.size());
//                assert(v1 < (int)P.size());
//                assert(v2 < (int)P.size());
//
//                mesh->add_triangle(v0, v1, v2, shader, smooth);
//            }
//
//            index_offset += nverts[i];
//        }
//
//        /* Vertex normals */
//        if (scene_write_xml_float3_array(VN, node, Attribute::standard_name(ATTR_STD_VERTEX_NORMAL))) {
//            Attribute* attr = mesh->attributes.add(ATTR_STD_VERTEX_NORMAL);
//            float3* fdata = attr->data_float3();
//
//            /* Loop over the normals */
//            for (auto n : VN) {
//                fdata[0] = n;
//                fdata++;
//            }
//        }
//
//        /* UV map */
//        if (scene_write_xml_float_array(UV, node, "UV") ||
//            scene_write_xml_float_array(UV, node, Attribute::standard_name(ATTR_STD_UV)))
//        {
//            Attribute* attr = mesh->attributes.add(ATTR_STD_UV);
//            float2* fdata = attr->data_float2();
//
//            /* Loop over the triangles */
//            index_offset = 0;
//            for (size_t i = 0; i < nverts.size(); i++) {
//                for (int j = 0; j < nverts[i] - 2; j++) {
//                    int v0 = index_offset;
//                    int v1 = index_offset + j + 1;
//                    int v2 = index_offset + j + 2;
//
//                    assert(v0 * 2 + 1 < (int)UV.size());
//                    assert(v1 * 2 + 1 < (int)UV.size());
//                    assert(v2 * 2 + 1 < (int)UV.size());
//
//                    fdata[0] = make_float2(UV[v0 * 2], UV[v0 * 2 + 1]);
//                    fdata[1] = make_float2(UV[v1 * 2], UV[v1 * 2 + 1]);
//                    fdata[2] = make_float2(UV[v2 * 2], UV[v2 * 2 + 1]);
//                    fdata += 3;
//                }
//
//                index_offset += nverts[i];
//            }
//        }
//
//        /* Tangents */
//        if (scene_write_xml_float_array(T, node, Attribute::standard_name(ATTR_STD_UV_TANGENT))) {
//            Attribute* attr = mesh->attributes.add(ATTR_STD_UV_TANGENT);
//            float3* fdata = attr->data_float3();
//
//            /* Loop over the triangles */
//            index_offset = 0;
//            for (size_t i = 0; i < nverts.size(); i++) {
//                for (int j = 0; j < nverts[i] - 2; j++) {
//                    int v0 = index_offset;
//                    int v1 = index_offset + j + 1;
//                    int v2 = index_offset + j + 2;
//
//                    assert(v0 * 3 + 2 < (int)T.size());
//                    assert(v1 * 3 + 2 < (int)T.size());
//                    assert(v2 * 3 + 2 < (int)T.size());
//
//                    fdata[0] = make_float3(T[v0 * 3], T[v0 * 3 + 1], T[v0 * 3 + 2]);
//                    fdata[1] = make_float3(T[v1 * 3], T[v1 * 3 + 1], T[v1 * 3 + 2]);
//                    fdata[2] = make_float3(T[v2 * 3], T[v2 * 3 + 1], T[v2 * 3 + 2]);
//                    fdata += 3;
//                }
//                index_offset += nverts[i];
//            }
//        }
//
//        /* Tangent signs */
//        if (scene_write_xml_float_array(TS, node, Attribute::standard_name(ATTR_STD_UV_TANGENT_SIGN))) {
//            Attribute* attr = mesh->attributes.add(ATTR_STD_UV_TANGENT_SIGN);
//            float* fdata = attr->data_float();
//
//            /* Loop over the triangles */
//            index_offset = 0;
//            for (size_t i = 0; i < nverts.size(); i++) {
//                for (int j = 0; j < nverts[i] - 2; j++) {
//                    int v0 = index_offset;
//                    int v1 = index_offset + j + 1;
//                    int v2 = index_offset + j + 2;
//
//                    assert(v0 < (int)TS.size());
//                    assert(v1 < (int)TS.size());
//                    assert(v2 < (int)TS.size());
//
//                    fdata[0] = TS[v0];
//                    fdata[1] = TS[v1];
//                    fdata[2] = TS[v2];
//                    fdata += 3;
//                }
//                index_offset += nverts[i];
//            }
//        }
//    }
//    else {
//        /* create vertices */
//        mesh->set_verts(P_array);
//
//        size_t num_ngons = 0;
//        size_t num_corners = 0;
//        for (size_t i = 0; i < nverts.size(); i++) {
//            num_ngons += (nverts[i] == 4) ? 0 : 1;
//            num_corners += nverts[i];
//        }
//        mesh->reserve_subd_faces(nverts.size(), num_ngons, num_corners);
//
//        /* create subd_faces */
//        int index_offset = 0;
//
//        for (size_t i = 0; i < nverts.size(); i++) {
//            mesh->add_subd_face(&verts[index_offset], nverts[i], shader, smooth);
//            index_offset += nverts[i];
//        }
//
//        /* UV map */
//        if (scene_write_xml_float_array(UV, node, "UV") ||
//            scene_write_xml_float_array(UV, node, Attribute::standard_name(ATTR_STD_UV)))
//        {
//            Attribute* attr = mesh->subd_attributes.add(ATTR_STD_UV);
//            float3* fdata = attr->data_float3();
//
//#if 0
//            if (subdivide_uvs) {
//                attr->flags |= ATTR_SUBDIVIDED;
//            }
//#endif
//
//            index_offset = 0;
//            for (size_t i = 0; i < nverts.size(); i++) {
//                for (int j = 0; j < nverts[i]; j++) {
//                    *(fdata++) = make_float3(UV[index_offset++]);
//                }
//            }
//        }
//
//        /* setup subd params */
//        float dicing_rate = state.dicing_rate;
//        scene_write_xml_float(&dicing_rate, node, "dicing_rate");
//        dicing_rate = std::max(0.1f, dicing_rate);
//
//        mesh->set_subd_dicing_rate(dicing_rate);
//        mesh->set_subd_objecttoworld(state.tfm);
//    }
//
//    /* we don't yet support arbitrary attributes, for now add vertex
//     * coordinates as generated coordinates if requested */
//    if (mesh->need_attribute(state.scene, ATTR_STD_GENERATED)) {
//        Attribute* attr = mesh->attributes.add(ATTR_STD_GENERATED);
//        memcpy(
//            attr->data_float3(), mesh->get_verts().data(), sizeof(float3) * mesh->get_verts().size());
//    }
}

/* Light */

//void scene_write_xml_light(XMLWriteState& state, xml_node node)
//{
//	//Light* light = new Light();
//
//	//light->set_shader(state.shader);
//	for(Light *light: state.scene->lights) {
//		//xml_node xml_node = node.append_child(light->type->name.c_str());
//		//if (light->get_light_type() != LIGHT_BACKGROUND)
//		xml_write_node(state, light, node);
//	}
//
//	//state.scene->lights.push_back(light);
//}

//void scene_write_xml_particle_systems(XMLWriteState& state, xml_node node)
//{
//	for(ParticleSystem * ps: state.scene->particle_systems) {
//		xml_write_node(state, ps, node);
//	}
//}

//void scene_write_xml_passes(XMLWriteState& state, xml_node node)
//{
//	foreach(Pass * pass, scene->passes) {
//		xml_write_node(pass, node);
//	}
//}
//
//void scene_write_xml_procedurals(XMLWriteState& state, xml_node node)
//{
//	foreach(Procedural * p, scene->procedurals) {
//		xml_write_node(p, node);
//	}
//}

/* Transform */

//void scene_write_xml_transform(xml_node node, Transform& tfm)
//{
//    if (node.append_attribute("matrix")) {
//        vector<float> matrix;
//        if (scene_write_xml_float_array(matrix, node, "matrix") && matrix.size() == 16) {
//            ProjectionTransform projection = *(ProjectionTransform*)&matrix[0];
//            tfm = tfm * projection_to_transform(projection_transpose(projection));
//        }
//    }
//
//    if (node.append_attribute("translate")) {
//        float3 translate = zero_float3();
//        scene_write_xml_float3(&translate, node, "translate");
//        tfm = tfm * transform_translate(translate);
//    }
//
//    if (node.append_attribute("rotate")) {
//        float4 rotate = zero_float4();
//        scene_write_xml_float4(&rotate, node, "rotate");
//        tfm = tfm * transform_rotate(DEG2RADF(rotate.x), make_float3(rotate.y, rotate.z, rotate.w));
//    }
//
//    if (node.append_attribute("scale")) {
//        float3 scale = zero_float3();
//        scene_write_xml_float3(&scale, node, "scale");
//        tfm = tfm * transform_scale(scale);
//    }
//}

/* State */

//void scene_write_xml_state(XMLReadState& state, xml_node node)
//{
//	/* Read shader */
//	string shadername;
//
//	if (scene_write_xml_string(&shadername, node, "shader")) {
//		bool found = false;
//
//		foreach(Shader * shader, state.scene->shaders) {
//			if (shader->name == shadername) {
//				state.shader = shader;
//				found = true;
//				break;
//			}
//		}
//
//		if (!found) {
//			fprintf(stderr, "Unknown shader \"%s\".\n", shadername.c_str());
//		}
//	}
//
//	/* Read object */
//	string objectname;
//
//	if (scene_write_xml_string(&objectname, node, "object")) {
//		bool found = false;
//
//		foreach(Object * object, state.scene->objects) {
//			if (object->name == objectname) {
//				state.object = object;
//				found = true;
//				break;
//			}
//		}
//
//		if (!found) {
//			fprintf(stderr, "Unknown object \"%s\".\n", objectname.c_str());
//		}
//	}
//
//	scene_write_xml_float(&state.dicing_rate, node, "dicing_rate");
//
//	/* read smooth/flat */
//	if (scene_write_xml_equal_string(node, "interpolation", "smooth")) {
//		state.smooth = true;
//	}
//	else if (scene_write_xml_equal_string(node, "interpolation", "flat")) {
//		state.smooth = false;
//	}
//}

/* Object */

void scene_write_xml_object(XMLWriteState& state, xml_node node)
{
//    Scene* scene = state.scene;
//
//    /* create mesh */
//    Mesh* mesh = new Mesh();
//    scene->geometry.push_back(mesh);
//
//    /* create object */
//    Object* object = new Object();
//    object->set_geometry(mesh);
//    object->set_tfm(state.tfm);
//
//    xml_write_node(state, object, node);
//
//    scene->objects.push_back(object);

	for(Object * object: state.scene->objects) {
		xml_node xml_node_obj = xml_write_node(state, object, node);

		if (object->attributes.size() > 0) {
			//AttributeSet attributes;
			//xml_node node_attributes = xml_node_obj.append_child("attributes");
			for(ParamValue & param_value: object->attributes) {
				xml_node node_attribute = xml_node_obj.append_child("attribute");

				//ustring m_name;   ///< data name
				xml_attribute attr_name = node_attribute.append_attribute("name");
				attr_name = (xml_pointer_to_name(&param_value) + "_" + param_value.name().string()).c_str();

				//TypeDesc m_type;  ///< data type, which may itself be an array
				//xml_attribute attr_type = node_attribute.append_attribute("type");
				//attr_type = (int)param_value.type();// .c_str();

				//ADD_ATTR(type);
				ADD_ATTR_DTYPE(param_value.type(), basetype);      ///< C data type at the heart of our type
				ADD_ATTR_DTYPE(param_value.type(), aggregate);     ///< What kind of AGGREGATE is it?
				ADD_ATTR_DTYPE(param_value.type(), vecsemantics);  ///< Hint: What does the aggregate represent?
				ADD_ATTR_DTYPE(param_value.type(), reserved);      ///< Reserved for future expansion
				ADD_ATTR_DTYPE(param_value.type(), arraylen);      ///< Array length, 0 = not array, -1 = unsized

				//union {
				//	char localval[16];
				//	const void* ptr;
				//} m_data;  ///< Our data, either a pointer or small local value

				vector<char> data(param_value.datasize());
				memcpy(data.data(), param_value.data(), param_value.datasize());
				xml_attribute attr_data = node_attribute.append_attribute("data");
				std::stringstream ss;
				ss << write_vector_to_binary_file(state, data);
				attr_data = ss.str().c_str();

				//int m_nvalues = 0;  ///< number of values of the given type
				//unsigned char m_interp = INTERP_CONSTANT;  ///< Interpolation type
				xml_attribute attr_interp = node_attribute.append_attribute("interp");
				attr_interp = param_value.interp();
				//bool m_copy = false;
				//bool m_nonlocal = false;
				//xml_attribute attr_nonlocal = node_attribute.append_attribute("nonlocal");
				//attr_nonlocal = param_value.is_nonlocal();
			}
		}
	}
}

/* Scene */

void scene_write_xml_scene(XMLWriteState& state, xml_node scene_node)
{
	///* default shaders */
	//Shader* default_surface;
	//Shader* default_volume;
	//Shader* default_light;
	//Shader* default_background;
	//Shader* default_empty;

	//unique_ptr_vector<Shader> shaders;
	for (Shader* shader : state.scene->shaders) {
		xml_node xml_shader = xml_write_node(state, shader, scene_node);
		scene_write_xml_shader_graph(state, shader, xml_shader);
	}

	//Camera* camera;
	xml_write_node(state, state.scene->camera, scene_node).attribute("name") = "camera";
	//Camera* dicing_camera;
	xml_write_node(state, state.scene->dicing_camera, scene_node).attribute("name") = "dicing_camera";
	//Film* film;
	xml_write_node(state, state.scene->film, scene_node).attribute("name") = "film";
	//Background* background;
	xml_write_node(state, state.scene->background, scene_node).attribute("name") = "background";
	//Integrator* integrator;
	xml_write_node(state, state.scene->integrator, scene_node).attribute("name") = "integrator";

	///* data lists */
	//unique_ptr_vector<Background> backgrounds;
	for (Background* b : state.scene->backgrounds) {
		xml_write_node(state, b, scene_node);
	}
	//unique_ptr_vector<Film> films;
	for (Film* f : state.scene->films) {
		xml_write_node(state, f, scene_node);
	}	 
	//unique_ptr_vector<Integrator> integrators;
	for (Integrator* i : state.scene->integrators) {
		xml_write_node(state, i, scene_node);
	}
	//unique_ptr_vector<Camera> cameras;
	for (Camera* c : state.scene->cameras) {
		xml_write_node(state, c, scene_node);
	}

	//unique_ptr_vector<Pass> passes;
	for (Pass* p : state.scene->passes) {
		xml_write_node(state, p, scene_node);
	}
	//unique_ptr_vector<ParticleSystem> particle_systems;
	for (ParticleSystem* ps : state.scene->particle_systems) {
		xml_write_node(state, ps, scene_node);
	}
	//unique_ptr_vector<Geometry> geometry;
	scene_write_xml_geom(state, scene_node);

	//unique_ptr_vector<Procedural> procedurals;
	for (Procedural* p : state.scene->procedurals) {
		//xml_write_node(state, p, scene_node); //TODO
	}

	//unique_ptr_vector<Object> objects;
	scene_write_xml_object(state, scene_node);

//	//for (xml_node node = scene_node.first_child(); node; node = node.next_sibling()) {
//		//if (string_iequals(node.name(), "film")) {
//	//xml_node film = scene_node.append_child(scene->film->type->name.c_str());
//	xml_write_node(state, state.scene->film, scene_node).attribute("name") = "film";
//	//}
////	else if (string_iequals(node.name(), "integrator")) {
//	//xml_node integrator = scene_node.append_child("integrator");
//	xml_write_node(state, state.scene->integrator, scene_node).attribute("name") = "integrator";
//	//}
//	//else if (string_iequals(node.name(), "camera")) {
//	//xml_node camera = scene_node.append_child("camera");
//	xml_write_node(state, state.scene->camera, scene_node).attribute("name") = "camera";
//	//}
//	//		else if (string_iequals(node.name(), "background")) {
//	//xml_node background = scene_node.append_child("background");
//	//scene_write_xml_background(scene, background);
//	xml_write_node(state, state.scene->background, scene_node).attribute("name") = "background";
	//		}
	//		else if (string_iequals(node.name(), "shader")) {
	//xml_node shader = scene_node.append_child("shader");
	//scene_write_xml_shader(state, scene_node);
	//		}
	//		else if (string_iequals(node.name(), "light")) {	
	//scene_write_xml_light(state, scene_node);
	//		}
	//		else if (string_iequals(node.name(), "mesh")) {
	//xml_node mesh = scene_node.append_child("mesh");
	//scene_write_xml_geom(state, scene_node);

	//scene_write_xml_particle_systems(state, scene_node);
	//scene_write_xml_passes(scene, scene_node);
	//scene_write_xml_procedurals(scene, scene_node);

	//		}
	//		else if (string_iequals(node.name(), "transform")) {
	//			XMLReadState substate = state;
	//
	   //         xml_node transform = scene_node.append_child("transform");
				//scene_write_xml_transform(transform, scene/*substate.tfm*/);
				//scene_write_xml_scene(scene, transform);
	//		}
	//		else if (string_iequals(node.name(), "state")) {
	//			XMLReadState substate = state;
	//
	   //         xml_node state = scene_node.append_child("state");
				//scene_write_xml_state(scene, state);
				//scene_write_xml_scene(scene, state);
	//		}
	//		else if (string_iequals(node.name(), "include")) {
	//			string src;
				//xml_node node_include = scene_node.append_child("include");

	//			if (scene_write_xml_string(&src, node, "src")) {
	//				scene_write_xml_include(state, src);
	//			}
	//		}
	//		else if (string_iequals(node.name(), "object")) {
	//			XMLReadState substate = state;
	//
	   //         xml_node object = scene_node.append_child("object");
	//scene_write_xml_object(state, scene_node);
				//scene_write_xml_scene(scene, object);
	//		}
	//#ifdef WITH_ALEMBIC
	//		else if (string_iequals(node.name(), "alembic")) {
	//			scene_write_xml_alembic(state, node);
	//		}
	//#endif
	//		else {
	//			fprintf(stderr, "Unknown node \"%s\".\n", node.name());
	//		}
		//}
}

/* Include */

void scene_write_xml_include(XMLWriteState &state, const string& filename_xml)
{
	/* open XML document */
	xml_document doc;
	//xml_parse_result parse_result;

	//string path = path_join(state.base, src);
	//parse_result = doc.load_file(path.c_str());

	//if (parse_result) {
		//XMLReadState substate = state;
		//substate.base = path_dirname(path);

	string filename_bin = string(filename_xml) + string(".bin");

	// Open the file in binary write mode
	state.file.open(filename_bin, std::ios::binary);
	if (!state.file.is_open()) {
		std::cerr << "Error: Could not open file for writing: " << filename_bin << "\n";
		return;
	}

	xml_node cycles = doc.append_child("cycles");
	scene_write_xml_scene(state, cycles);
	//}
	//else {
	//    fprintf(stderr, "%s read error: %s\n", src.c_str(), parse_result.description());
	//    exit(EXIT_FAILURE);
	//}

	// Save the XML to a file
	doc.save_file(filename_xml.c_str());

	state.file.close();
}

/* File */

void scene_write_xml_file(Scene* scene, const char* filepath)
{
	if (scene == nullptr || filepath == nullptr)
		return;

	XMLWriteState state;

	state.scene = scene;
	//std::string base = path_dirname(filepath);

	scene_write_xml_include(state, filepath);	
}

CCL_NAMESPACE_END

