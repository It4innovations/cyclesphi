/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/unique_ptr.h"
#include "util/vector.h"
#include <functional>
#include <string>

CCL_NAMESPACE_BEGIN

class Scene;
class Shader;
class ShaderGraph;

void xml_read_file(Scene *scene, const char *filepath);
void xml_set_volume_to_attr(Scene* scene, std::string geom_name, std::string attr_name, int type, vector<char>& file_content);
void xml_set_material_to_node(Scene* scene, const char* file_content);
void xml_set_material_to_shader(Scene* scene, const char* file_content);
void xml_set_material_to_shader2(Scene* scene, Shader* shader, const char* file_content);

/* Read a shader graph from the text of a <shader> element. There is no binary sidecar file:
 * arrays are read from the text. The sockets of the <shader> element are set on `shader`, the
 * graph is returned without being set on it, so that the caller can still edit the graph.
 * Returns nullptr when the text is not a <shader> document.
 *
 * Warnings and errors are passed to `report` (printed to stderr when it is not set). `error` is
 * set for problems which make the graph unusable (an unknown node type, a connection to an
 * unknown node, a cyclic connection), and not set for ignored values and links. */
unique_ptr<ShaderGraph> xml_read_shader_graph_from_string(
    Scene *scene,
    Shader *shader,
    const char *xml_text,
    size_t xml_size,
    const std::function<void(bool error, const std::string &message)> &report = nullptr);

/* macros for importing */
#define RAD2DEGF(_rad) ((_rad) * (float)(180.0 / M_PI))
#define DEG2RADF(_deg) ((_deg) * (float)(M_PI / 180.0))

CCL_NAMESPACE_END
