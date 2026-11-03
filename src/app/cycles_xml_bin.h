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

/* Run time configuration of the neural volume decoder (.srnxr format 2), set from the command
 * line before reading the scene. Non-empty values override the srnxr_* attributes of the XML
 * <attribute> elements. */
struct XMLNeuralVolumeOptions {
  /* cuda_fused, cuda_tensor or coopvec */
  std::string backend;
  /* accurate (sinf) or fast (__sinf on CUDA) */
  std::string sine;
  /* Backend used, with a warning, when the requested one cannot run on the render device.
   * Empty: such volumes are not loaded. */
  std::string fallback;
  /* Benchmark mode: never fall back, a backend the device cannot run is an error. */
  bool benchmark = false;
  /* -1: from XML (default on), 0: off, 1: on. Device statistics of the decoder. */
  int counters = -1;
  /* -1: from XML (default on), 0: off, 1: on. Cooperative decoding of partially active warps. */
  int partial_warps = -1;
  /* certified (default) or sampled (not certified), empty: from XML. Majorant bake. */
  std::string majorant;
  /* -1: from XML (default off), 0: off, 1: on. Wavefront decode queue (record the lookups of
   * the volume walks, decode them in batches on Tensor Cores, replay). */
  int queue = -1;
  /* Lookups per path and launch kept by the queue (0: default). */
  int queue_window = 0;
  /* Requests per launch (0: 4 x the paths). */
  int queue_capacity = 0;
};

/* Apply the integrator options of the command line to the scene (after reading the file). */
void xml_apply_neural_volume_options(Scene *scene);
void xml_set_neural_volume_options(const XMLNeuralVolumeOptions &options);
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
