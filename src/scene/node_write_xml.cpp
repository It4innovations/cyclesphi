/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/node_write_xml.h"

//#include "util/foreach.h"
#include "util/string.h"
#include "util/transform.h"
#include "util/path.h"

#include "scene/object.h"

CCL_NAMESPACE_BEGIN

// Function to write an array to a binary file
template<typename T>
string write_array_to_binary_file(XMLWriter& writer, const array<T>& data)
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

bool xml_pointer_to_name_check(Node* node) {
    if (node->name == "default_background" ||
        node->name == "default_empty" ||
        node->name == "default_light" ||
        node->name == "default_surface" ||
        node->name == "default_volume") {

        return false;
    }

    return true;
}
string xml_pointer_to_name(void *ptr) 
{
    // Convert void* to name
    size_t ptrAsSizeT = reinterpret_cast<size_t>(ptr);
    std::string ptrAsString = std::to_string(ptrAsSizeT);
    return ptrAsString;
}

string xml_fix_tag(ustring tag)
{
    string str = tag.string();
    std::replace(str.begin(), str.end(), ' ', '_');

    return str;
}

//static bool xml_read_boolean(const char *value)
//{
//  return string_iequals(value, "true") || (atoi(value) != 0);
//}

static const char *xml_write_boolean(bool value)
{
  return (value) ? "true" : "false";
}

//template<int VECTOR_SIZE, typename T>
//static void xml_read_float_array(T &value, xml_attribute attr)
//{
//  vector<string> tokens;
//  string_split(tokens, attr.value());
//
//  if (tokens.size() % VECTOR_SIZE != 0) {
//    return;
//  }
//
//  value.resize(tokens.size() / VECTOR_SIZE);
//  for (size_t i = 0; i < value.size(); i++) {
//    float *value_float = (float *)&value[i];
//
//    for (size_t j = 0; j < VECTOR_SIZE; j++) {
//      value_float[j] = (float)atof(tokens[i * VECTOR_SIZE + j].c_str());
//    }
//  }
//}

//void xml_read_node(XMLReader &reader, Node *node, xml_node xml_node)
//{
//  xml_attribute name_attr = xml_node.attribute("name");
//  if (name_attr) {
//    node->name = ustring(name_attr.value());
//  }
//
//  foreach (const SocketType &socket, node->type->inputs) {
//    if (socket.type == SocketType::CLOSURE || socket.type == SocketType::UNDEFINED) {
//      continue;
//    }
//    if (socket.flags & SocketType::INTERNAL) {
//      continue;
//    }
//
//    xml_attribute attr = xml_node.attribute(socket.ui_name.c_str());
//
//    if (!attr) {
//      continue;
//    }
//
//    switch (socket.type) {
//      case SocketType::BOOLEAN: {
//        node->set(socket, xml_read_boolean(attr.value()));
//        break;
//      }
//      case SocketType::BOOLEAN_ARRAY: {
//        vector<string> tokens;
//        string_split(tokens, attr.value());
//
//        array<bool> value;
//        value.resize(tokens.size());
//        for (size_t i = 0; i < value.size(); i++) {
//          value[i] = xml_read_boolean(tokens[i].c_str());
//        }
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::FLOAT: {
//        node->set(socket, (float)atof(attr.value()));
//        break;
//      }
//      case SocketType::FLOAT_ARRAY: {
//        array<float> value;
//        xml_read_float_array<1>(value, attr);
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::INT: {
//        node->set(socket, (int)atoi(attr.value()));
//        break;
//      }
//      case SocketType::UINT: {
//        node->set(socket, (uint)atoi(attr.value()));
//        break;
//      }
//      case SocketType::UINT64: {
//        node->set(socket, (uint64_t)strtoull(attr.value(), nullptr, 10));
//        break;
//      }
//      case SocketType::INT_ARRAY: {
//        vector<string> tokens;
//        string_split(tokens, attr.value());
//
//        array<int> value;
//        value.resize(tokens.size());
//        for (size_t i = 0; i < value.size(); i++) {
//          value[i] = (int)atoi(attr.value());
//        }
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::COLOR:
//      case SocketType::VECTOR:
//      case SocketType::POINT:
//      case SocketType::NORMAL: {
//        array<float3> value;
//        xml_read_float_array<3>(value, attr);
//        if (value.size() == 1) {
//          node->set(socket, value[0]);
//        }
//        break;
//      }
//      case SocketType::COLOR_ARRAY:
//      case SocketType::VECTOR_ARRAY:
//      case SocketType::POINT_ARRAY:
//      case SocketType::NORMAL_ARRAY: {
//        array<float3> value;
//        xml_read_float_array<3>(value, attr);
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::POINT2: {
//        array<float2> value;
//        xml_read_float_array<2>(value, attr);
//        if (value.size() == 1) {
//          node->set(socket, value[0]);
//        }
//        break;
//      }
//      case SocketType::POINT2_ARRAY: {
//        array<float2> value;
//        xml_read_float_array<2>(value, attr);
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::STRING: {
//        node->set(socket, attr.value());
//        break;
//      }
//      case SocketType::ENUM: {
//        ustring value(attr.value());
//        if (socket.enum_values->exists(value)) {
//          node->set(socket, value);
//        }
//        else {
//          fprintf(stderr,
//                  "Unknown value \"%s\" for attribute \"%s\".\n",
//                  value.c_str(),
//                  socket.ui_name.c_str());
//        }
//        break;
//      }
//      case SocketType::STRING_ARRAY: {
//        vector<string> tokens;
//        string_split(tokens, attr.value());
//
//        array<ustring> value;
//        value.resize(tokens.size());
//        for (size_t i = 0; i < value.size(); i++) {
//          value[i] = ustring(tokens[i]);
//        }
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::TRANSFORM: {
//        array<Transform> value;
//        xml_read_float_array<12>(value, attr);
//        if (value.size() == 1) {
//          node->set(socket, value[0]);
//        }
//        break;
//      }
//      case SocketType::TRANSFORM_ARRAY: {
//        array<Transform> value;
//        xml_read_float_array<12>(value, attr);
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::NODE: {
//        ustring value(attr.value());
//        map<ustring, Node *>::iterator it = reader.node_map.find(value);
//        if (it != reader.node_map.end()) {
//          Node *value_node = it->second;
//          if (value_node->is_a(socket.node_type)) {
//            node->set(socket, it->second);
//          }
//        }
//        break;
//      }
//      case SocketType::NODE_ARRAY: {
//        vector<string> tokens;
//        string_split(tokens, attr.value());
//
//        array<Node *> value;
//        value.resize(tokens.size());
//        for (size_t i = 0; i < value.size(); i++) {
//          map<ustring, Node *>::iterator it = reader.node_map.find(ustring(tokens[i]));
//          if (it != reader.node_map.end()) {
//            Node *value_node = it->second;
//            value[i] = (value_node->is_a(socket.node_type)) ? value_node : NULL;
//          }
//          else {
//            value[i] = NULL;
//          }
//        }
//        node->set(socket, value);
//        break;
//      }
//      case SocketType::CLOSURE:
//      case SocketType::UNDEFINED:
//      case SocketType::NUM_TYPES:
//        break;
//    }
//  }
//
//  if (!node->name.empty()) {
//    reader.node_map[node->name] = node;
//  }
//}

void xml_write_node_socket(XMLWriter& writer, Node* node, xml_node xml_root, const SocketType& socket, xml_node xml_node_main, string xml_node_name)
{
    if (socket.type == SocketType::CLOSURE || socket.type == SocketType::UNDEFINED) {
        return;
    }
    if (socket.flags & SocketType::INTERNAL) {
        return;
    }
    if (socket.default_value && node->has_default_value(socket)) {
        return;
    }

    string attr;

    switch (socket.type) {
    case SocketType::BOOLEAN: {
        attr = xml_write_boolean(node->get_bool(socket));
        break;
    }
    case SocketType::BOOLEAN_ARRAY: {
        std::stringstream ss;
        const array<bool>& value = node->get_bool_array(socket);
        ss << write_array_to_binary_file(writer, value);
        attr = ss.str();
        break;
    }
    case SocketType::FLOAT: {
        std::stringstream ss;
        ss << (double)node->get_float(socket);
        attr = ss.str();
        break;
    }
    case SocketType::FLOAT_ARRAY: {
        std::stringstream ss;
        const array<float>& value = node->get_float_array(socket);
        //for (size_t i = 0; i < value.size(); i++) {
        //  ss << value[i];
        //  if (i != value.size() - 1) {
        //    ss << " ";
        //  }
        //}
        ss << write_array_to_binary_file(writer, value);
        attr = ss.str();
        break;
    }
    case SocketType::INT: {
        std::stringstream ss;
        ss << node->get_int(socket);
        attr = ss.str();
        break;
    }
    case SocketType::UINT: {
        std::stringstream ss;
        ss << node->get_uint(socket);
        attr = ss.str();
        break;
    }
    case SocketType::UINT64: {
        std::stringstream ss;
        ss << node->get_uint64(socket);
        attr = ss.str();
        break;
    }
    case SocketType::INT_ARRAY: {
        std::stringstream ss;
        const array<int>& value = node->get_int_array(socket);
        //for (size_t i = 0; i < value.size(); i++) {
        //  ss << value[i];
        //  if (i != value.size() - 1) {
        //    ss << " ";
        //  }
        //}
        ss << write_array_to_binary_file(writer, value);
        attr = ss.str();
        break;
    }
    case SocketType::COLOR:
    case SocketType::VECTOR:
    case SocketType::POINT:
    case SocketType::NORMAL: {
        float3 value = node->get_float3(socket);
        attr =
            string_printf("%g %g %g %g", (double)value.x, (double)value.y, (double)value.z, (double)value.w);
        break;
    }
    case SocketType::COLOR_ARRAY:
    case SocketType::VECTOR_ARRAY:
    case SocketType::POINT_ARRAY:
    case SocketType::NORMAL_ARRAY: {
        std::stringstream ss;
        const array<float3>& value = node->get_float3_array(socket);
        //for (size_t i = 0; i < value.size(); i++) {
        //  ss << string_printf(
        //      "%g %g %g %g", (double)value[i].x, (double)value[i].y, (double)value[i].z, (double)value[i].w);
        //  if (i != value.size() - 1) {
        //    ss << " ";
        //  }
        //}
        ss << write_array_to_binary_file(writer, value);
        attr = ss.str();
        break;
    }
    case SocketType::POINT2: {
        float2 value = node->get_float2(socket);
        attr = string_printf("%g %g", (double)value.x, (double)value.y);
        break;
    }
    case SocketType::POINT2_ARRAY: {
        std::stringstream ss;
        const array<float2>& value = node->get_float2_array(socket);
        //for (size_t i = 0; i < value.size(); i++) {
        //  ss << string_printf("%g %g", (double)value[i].x, (double)value[i].y);
        //  if (i != value.size() - 1) {
        //    ss << " ";
        //  }
        //}
        ss << write_array_to_binary_file(writer, value);
        attr = ss.str();
        break;
    }
    case SocketType::ENUM:
    case SocketType::STRING: {
        std::stringstream ss;
        ss << node->get_string(socket);
        attr = ss.str();
        break;
    }
    //case SocketType::ENUM: {
    //    std::stringstream ss;
    //    ss << node->get_int(socket);
    //    attr = ss.str();
    //    break;
    //}
    case SocketType::STRING_ARRAY: {
        std::stringstream ss;
        const array<ustring>& value = node->get_string_array(socket);
        for (size_t i = 0; i < value.size(); i++) {
            ss << value[i];
            if (i != value.size() - 1) {
                ss << " ";
            }
        }
        attr = ss.str();
        break;
    }
    case SocketType::TRANSFORM: {
        Transform tfm = node->get_transform(socket);
        
        Object* ob = dynamic_cast<Object*>(node);
        if (!ob || ob->get_geometry() && (
            ob->get_geometry()->geometry_type == Geometry::Type::VOLUME
            || ob->get_geometry()->geometry_type == Geometry::Type::LIGHT
            || !ob->get_geometry()->transform_applied)) {
            std::stringstream ss;
            for (int i = 0; i < 3; i++) {
                if (i == 2)
                    ss << string_printf("%g %g %g %g",
                        (double)tfm[i][0],
                        (double)tfm[i][1],
                        (double)tfm[i][2],
                        (double)tfm[i][3]);
                else
                    ss << string_printf("%g %g %g %g ",
                        (double)tfm[i][0],
                        (double)tfm[i][1],
                        (double)tfm[i][2],
                        (double)tfm[i][3]);
            }
            //ss << string_printf("%g %g %g %g", 0.0, 0.0, 0.0, 1.0);
            attr = ss.str();
        }
        break;
    }
    case SocketType::TRANSFORM_ARRAY: {
        std::stringstream ss;
        const array<Transform>& value = node->get_transform_array(socket);
        //for (size_t j = 0; j < value.size(); j++) {
        //  const Transform &tfm = value[j];

        //  for (int i = 0; i < 3; i++) {
        //    if (i == 2)
        //      ss << string_printf("%g %g %g %g",
        //          (double)tfm[i][0],
        //          (double)tfm[i][1],
        //          (double)tfm[i][2],
        //          (double)tfm[i][3]);
        //    else
        //      ss << string_printf("%g %g %g %g ",
        //          (double)tfm[i][0],
        //          (double)tfm[i][1],
        //          (double)tfm[i][2],
        //          (double)tfm[i][3]);
        //  }
        //  //ss << string_printf("%g %g %g %g", 0.0, 0.0, 0.0, 1.0);
        //  if (j != value.size() - 1) {
        //    ss << " ";
        //  }
        //}
        ss << write_array_to_binary_file(writer, value);
        attr = ss.str();
        break;
    }
    case SocketType::NODE: {
        Node* value = node->get_node(socket);
        if (value) {
            if (xml_pointer_to_name_check(value))
                attr = xml_pointer_to_name(value);
            else
                attr = value->name.string();
        }
        break;
    }
    case SocketType::NODE_ARRAY: {
        std::stringstream ss;
        const array<Node*>& value = node->get_node_array(socket);
        for (size_t i = 0; i < value.size(); i++) {
            if (value[i]) {
                if (xml_pointer_to_name_check(value[i]))
                    ss << xml_pointer_to_name(value[i]);
                else
                    ss << value[i]->name;
            }
            if (i != value.size() - 1) {
                ss << " ";
            }
        }
        attr = ss.str();
        break;
    }
    case SocketType::CLOSURE:
    case SocketType::UNDEFINED:
    case SocketType::NUM_TYPES:
        break;
    }

    if (attr.empty())
        return;

    string socket_file = xml_node_name + "_" + socket.name.string();

    xml_node xml_node_socket = xml_node_main.append_child("socket");

    xml_attribute attr_name = xml_node_socket.append_attribute("name");
    attr_name = socket.name.c_str();

    xml_attribute attr_ui_name = xml_node_socket.append_attribute("ui_name");
    attr_ui_name = socket.ui_name.c_str();

    xml_attribute attr_value = xml_node_socket.append_attribute("value");
    attr_value = attr.c_str();

    xml_attribute attr_type_name = xml_node_socket.append_attribute("type_name");
    attr_type_name = SocketType::type_name(socket.type).c_str();
}

xml_node xml_write_node(XMLWriter& writer, Node* node, xml_node xml_root)
{
  xml_node xml_node_main = xml_root.append_child(node->type->name.c_str());

  string xml_node_name = node->name.string();
  if (xml_pointer_to_name_check(node))
      xml_node_name = xml_pointer_to_name(node);

  xml_node_main.append_attribute("name") = xml_node_name.c_str();
  xml_node_main.append_attribute("ui_name") = node->name.string().c_str();

  for (const SocketType &socket: node->type->inputs) {
      xml_write_node_socket(writer, node, xml_root, socket, xml_node_main, xml_node_name);
  }

  //foreach(const SocketType& socket, node->type->outputs) {
  //    xml_write_node_socket(writer, node, xml_root, socket, xml_node_main, xml_node_name);
  //}

  return xml_node_main;
}

CCL_NAMESPACE_END
