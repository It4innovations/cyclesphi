/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

//#include "graph/node_xml.h"
#include "scene/scene.h"

CCL_NAMESPACE_BEGIN

#if 0
void scene_write_xml_int(int value, xml_node node, const char* name);
void scene_write_xml_int_array(vector<int>& value, xml_node node, const char* name);
void scene_write_xml_float(float value, xml_node node, const char* name);
void scene_write_xml_float_array(vector<float>& value, xml_node node, const char* name);
void scene_write_xml_float3(float3 value, xml_node node, const char* name);
void scene_write_xml_float3_array(vector<float3>& value, xml_node node, const char* name);
void scene_write_xml_float4(float4 value, xml_node node, const char* name);
void scene_write_xml_string(string str, xml_node node, const char* name);
void scene_write_xml_camera(Scene *scene, xml_node node);
//void scene_write_xml_alembic(Scene *scene, xml_node graph_node);
void scene_write_xml_shader_graph(Scene *scene, Shader* shader, xml_node graph_node);
void scene_write_xml_shader(Scene *scene, xml_node node);
void scene_write_xml_background(Scene *scene, xml_node node);
//void scene_write_xml_mesh(const Scene *scene, xml_node node);
void scene_write_xml_light(Scene *scene, xml_node node);
void scene_write_xml_transform(xml_node node, Transform& tfm);
void scene_write_xml_state(Scene *scene, xml_node node);
void scene_write_xml_object(Scene *scene, xml_node node);
void scene_write_xml_include(Scene *scene, const string& src);
void scene_write_xml_scene(Scene *scene, xml_node scene_node);
void scene_write_xml_include(Scene *scene, const string& src);
#endif

void scene_write_xml_file(Scene* scene, const char* filepath);

CCL_NAMESPACE_END

