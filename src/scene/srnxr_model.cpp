/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/srnxr_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

CCL_NAMESPACE_BEGIN

// .srnxr file format version 2 (little-endian), see tools/srnxr/srnxr_format.py:
//   header (124 bytes, padded to 128), section table (64 bytes per section:
//   name[32], dtype, element_count, byte_offset, byte_length, crc32), 64-byte aligned sections.
namespace {

const char file_magic[8] = {'S', 'R', 'N', 'X', 'R', '\0', '\x01', '\0'};
const uint32_t file_version2 = 2;
const uint32_t endian_marker = 0x01020304u;
const size_t header_struct_size = 124;
const size_t header_crc_offset = header_struct_size - 8;
const size_t file_crc_offset = header_struct_size - 4;
const size_t section_entry_size = 64;

enum DType : uint32_t {
  DTYPE_FLOAT32 = 1,
  DTYPE_FLOAT16 = 2,
  DTYPE_UINT32 = 3,
  DTYPE_UINT8 = 4,
  DTYPE_JSON_UTF8 = 5,
};

enum Architecture : uint32_t {
  ARCH_SIREN = 1,
  ARCH_FOURIER_MLP = 2,
};

enum FileFlags : uint32_t {
  FILE_FLAG_LINEAR_HEAD = 1u << 0,
  FILE_FLAG_HAS_FP32_MASTER = 1u << 1,
};

struct Section {
  std::string name;
  uint32_t dtype;
  uint64_t element_count;
  uint64_t byte_offset;
  uint64_t byte_length;
};

template<typename T> T read_value(const char *data, const size_t offset)
{
  T value;
  memcpy(&value, data + offset, sizeof(T));
  return value;
}

uint32_t crc32_update(uint32_t crc, const uint8_t *data, const size_t size)
{
  static uint32_t table[256];
  static bool table_init = false;
  if (!table_init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int bit = 0; bit < 8; ++bit) {
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[i] = c;
    }
    table_init = true;
  }
  for (size_t i = 0; i < size; ++i) {
    crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc;
}

uint32_t crc32(const char *data, const size_t size)
{
  return crc32_update(0xFFFFFFFFu, (const uint8_t *)data, size) ^ 0xFFFFFFFFu;
}

/* IEEE half <-> float, exact for every half value, round to nearest even for float -> half. */
float half_bits_to_float(const uint16_t h)
{
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  const uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits;
  if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  }
  else if (exponent != 0) {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  else if (mantissa == 0) {
    bits = sign;
  }
  else {
    /* Subnormal half, normalize. */
    int e = -1;
    do {
      e++;
      mantissa <<= 1;
    } while ((mantissa & 0x400u) == 0);
    bits = sign | (uint32_t(112 - e) << 23) | ((mantissa & 0x3FFu) << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t float_to_half_bits(const float f)
{
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t abs = x & 0x7FFFFFFFu;
  if (abs >= 0x7F800000u) {
    return uint16_t(sign | 0x7C00u | ((abs > 0x7F800000u) ? 0x200u : 0u));
  }
  if (abs >= 0x477FF000u) {
    return uint16_t(sign | 0x7C00u); /* overflow to infinity (65520 and above) */
  }
  if (abs < 0x38800000u) {
    /* Subnormal or zero half. */
    if (abs < 0x33000000u) {
      return uint16_t(sign);
    }
    const uint32_t exponent = abs >> 23;
    const uint32_t mantissa = (abs & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126 - exponent;
    uint32_t result = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (result & 1u))) {
      result++;
    }
    return uint16_t(sign | result);
  }
  uint32_t result = ((abs >> 13) - (112u << 10));
  const uint32_t rest = abs & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (result & 1u))) {
    result++;
  }
  return uint16_t(sign | result);
}

const Section *find_section(const std::vector<Section> &sections, const char *name)
{
  for (const Section &section : sections) {
    if (section.name == name) {
      return &section;
    }
  }
  return nullptr;
}

struct Parser {
  const char *data;
  size_t size;
  std::vector<Section> sections;
  std::string &error;

  bool fail(const std::string &message)
  {
    error = message;
    return false;
  }

  const Section *section(const char *name, const uint64_t expected_count, const bool optional)
  {
    const Section *s = find_section(sections, name);
    if (!s) {
      if (!optional) {
        error = std::string("missing section ") + name;
      }
      return nullptr;
    }
    if (expected_count != 0 && s->element_count != expected_count) {
      char buf[256];
      snprintf(buf,
               sizeof(buf),
               "section %s has %llu elements, expected %llu",
               name,
               (unsigned long long)s->element_count,
               (unsigned long long)expected_count);
      error = buf;
      return nullptr;
    }
    return s;
  }

  bool floats(const char *name, const uint64_t expected_count, std::vector<float> &result)
  {
    const Section *s = section(name, expected_count, false);
    if (!s) {
      return false;
    }
    const char *base = data + s->byte_offset;
    result.resize(s->element_count);
    if (s->dtype == DTYPE_FLOAT32 && s->byte_length == s->element_count * sizeof(float)) {
      memcpy(result.data(), base, result.size() * sizeof(float));
    }
    else if (s->dtype == DTYPE_FLOAT16 && s->byte_length == s->element_count * sizeof(uint16_t))
    {
      for (size_t i = 0; i < result.size(); ++i) {
        result[i] = half_bits_to_float(read_value<uint16_t>(base, i * sizeof(uint16_t)));
      }
    }
    else {
      return fail(std::string("section ") + name + " is not a float section");
    }
    for (const float v : result) {
      if (!std::isfinite(v)) {
        return fail(std::string("section ") + name + " contains a non-finite value");
      }
    }
    return true;
  }

  bool uints(const char *name, const uint64_t expected_count, std::vector<uint32_t> &result)
  {
    const Section *s = section(name, expected_count, false);
    if (!s) {
      return false;
    }
    if (s->dtype != DTYPE_UINT32 || s->byte_length != s->element_count * sizeof(uint32_t)) {
      return fail(std::string("section ") + name + " is not a uint32 section");
    }
    result.resize(s->element_count);
    memcpy(result.data(), data + s->byte_offset, result.size() * sizeof(uint32_t));
    return true;
  }

  bool bytes(const char *name, const bool optional, std::string &result)
  {
    const Section *s = section(name, 0, optional);
    if (!s) {
      return optional && error.empty();
    }
    if ((s->dtype != DTYPE_UINT8 && s->dtype != DTYPE_JSON_UTF8) ||
        s->byte_length != s->element_count)
    {
      return fail(std::string("section ") + name + " is not a byte section");
    }
    result.assign(data + s->byte_offset, s->byte_length);
    return true;
  }
};

template<typename T> uint64_t append_array(std::vector<char> &blob, const T *values, size_t count)
{
  const uint64_t offset = (blob.size() + SRNXR_DEVICE_ALIGNMENT - 1) /
                          SRNXR_DEVICE_ALIGNMENT * SRNXR_DEVICE_ALIGNMENT;
  blob.resize(offset + count * sizeof(T), 0);
  if (count) {
    memcpy(blob.data() + offset, values, count * sizeof(T));
  }
  return offset;
}

bool box_intersects(const float *amin, const float *amax, const float *bmin, const float *bmax, int dims)
{
  for (int d = 0; d < dims; d++) {
    if (amin[d] > bmax[d] || bmin[d] > amax[d]) {
      return false;
    }
  }
  return true;
}

double seconds_since(const std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

const char *srnxr_backend_name(const SRNXRBackend backend)
{
  switch (backend) {
    case SRNXR_BACKEND_CUDA_FUSED:
      return "CUDA_FUSED";
    case SRNXR_BACKEND_CUDA_TENSOR:
      return "CUDA_TENSOR";
    case SRNXR_BACKEND_COOPVEC:
      return "COOPVEC";
    case SRNXR_BACKEND_NUM:
      break;
  }
  return "UNKNOWN";
}

bool srnxr_backend_from_name(const std::string &name, SRNXRBackend &backend)
{
  std::string n = name;
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::toupper(c); });
  for (int i = 0; i < SRNXR_BACKEND_NUM; i++) {
    if (n == srnxr_backend_name(SRNXRBackend(i))) {
      backend = SRNXRBackend(i);
      return true;
    }
  }
  return false;
}

const char *srnxr_sine_mode_name(const SRNXRSineMode mode)
{
  return (mode == SRNXR_SINE_FAST) ? "fast" : "accurate";
}

bool srnxr_sine_mode_from_name(const std::string &name, SRNXRSineMode &mode)
{
  if (name == "accurate" || name == "sinf") {
    mode = SRNXR_SINE_ACCURATE;
    return true;
  }
  if (name == "fast" || name == "__sinf") {
    mode = SRNXR_SINE_FAST;
    return true;
  }
  return false;
}

std::string SRNXRBuildReport::summary() const
{
  std::string s;
  char buf[512];
  snprintf(buf,
           sizeof(buf),
           "  Format: %u, SIREN %u -> %u x %u -> %u, w0:",
           format_version,
           input_dim,
           hidden_layer_count,
           hidden_width,
           output_dim);
  s += buf;
  for (const float w0 : layer_w0) {
    snprintf(buf, sizeof(buf), " %g", w0);
    s += buf;
  }
  s += "\n  Fields:";
  for (const std::string &name : field_names) {
    s += " " + name;
  }
  s += ", channels:";
  for (const uint32_t c : channel_field) {
    s += " " + field_names[c];
  }
  snprintf(buf,
           sizeof(buf),
           "\n  Octree: %u nodes, %u leaves, %u networks, depth %u\n"
           "  Neighbors per leaf: mean %.2f, max %u; expansion beyond the strict box: %.3f - %.3f\n"
           "  File: %zu bytes, device: %zu bytes (weights %zu, mma copy %zu, warp copy %zu)\n"
           "  Parse %.3f s, pack %.3f s%s%s\n",
           node_count,
           leaf_count,
           network_count,
           max_depth,
           mean_neighbors,
           max_neighbors,
           overlap_min,
           overlap_max,
           file_bytes,
           device_bytes,
           weight_bytes,
           mma_weight_bytes,
           warp_weight_bytes,
           parse_seconds,
           pack_seconds,
           synthetic ? "\n  Model metadata marks it SYNTHETIC" : "",
           weights_were_fp32 ? "\n  FP32 weights were quantized to FP16 by the loader" : "");
  s += buf;
  for (const std::string &w : warnings) {
    s += "  Warning: " + w + "\n";
  }
  return s;
}

bool srnxr_is_version2(const char *data, const size_t size)
{
  return size >= 16 && memcmp(data, file_magic, sizeof(file_magic)) == 0 &&
         read_value<uint32_t>(data, 12) == file_version2;
}

bool srnxr_build_device_blob_v2(const char *data,
                                const size_t size,
                                const SRNXRBuildSettings &settings,
                                std::vector<char> &blob,
                                SRNXRBuildReport &report,
                                std::string &error)
{
  const auto parse_start = std::chrono::steady_clock::now();
  blob.clear();
  error.clear();
  report = SRNXRBuildReport();
  report.file_bytes = size;

  Parser p{data, size, {}, error};

  /* Header */
  if (size < header_struct_size || memcmp(data, file_magic, sizeof(file_magic)) != 0) {
    return p.fail("not a .srnxr file");
  }
  const uint32_t header_size = read_value<uint32_t>(data, 8);
  report.format_version = read_value<uint32_t>(data, 12);
  if (read_value<uint32_t>(data, 16) != endian_marker) {
    return p.fail("endian mismatch");
  }
  if (report.format_version != file_version2 || header_size != header_struct_size) {
    return p.fail("unsupported format version or header size");
  }
  if (crc32(data, header_crc_offset) != read_value<uint32_t>(data, header_crc_offset)) {
    return p.fail("header checksum mismatch");
  }
  if (read_value<uint64_t>(data, 108) != size) {
    return p.fail("file size does not match the header (truncated file?)");
  }
  {
    const uint8_t zero[4] = {0, 0, 0, 0};
    uint32_t crc = crc32_update(0xFFFFFFFFu, (const uint8_t *)data, file_crc_offset);
    crc = crc32_update(crc, zero, sizeof(zero));
    crc = crc32_update(
        crc, (const uint8_t *)data + header_struct_size, size - header_struct_size);
    if ((crc ^ 0xFFFFFFFFu) != read_value<uint32_t>(data, file_crc_offset)) {
      return p.fail("file checksum mismatch");
    }
  }

  const uint32_t architecture = read_value<uint32_t>(data, 20);
  const uint32_t input_dim = read_value<uint32_t>(data, 24);
  const uint32_t output_dim = read_value<uint32_t>(data, 28);
  const uint32_t layers = read_value<uint32_t>(data, 32);
  const uint32_t hidden_width = read_value<uint32_t>(data, 36);
  const uint32_t network_count = read_value<uint32_t>(data, 40);
  const uint32_t leaf_count = read_value<uint32_t>(data, 44);
  const uint32_t node_count = read_value<uint32_t>(data, 48);
  const uint32_t weights_dtype = read_value<uint32_t>(data, 52);
  const uint32_t file_flags = read_value<uint32_t>(data, 56);
  float domain_min[4], domain_max[4];
  for (int d = 0; d < 4; d++) {
    domain_min[d] = read_value<float>(data, 60 + 4 * d);
    domain_max[d] = read_value<float>(data, 76 + 4 * d);
  }
  const uint64_t section_table_offset = read_value<uint64_t>(data, 92);
  const uint64_t section_count = read_value<uint64_t>(data, 100);

  report.input_dim = input_dim;
  report.output_dim = output_dim;
  report.hidden_layer_count = layers;
  report.hidden_width = hidden_width;
  report.network_count = network_count;
  report.leaf_count = leaf_count;
  report.node_count = node_count;
  report.has_fp32_master = (file_flags & FILE_FLAG_HAS_FP32_MASTER) != 0;

  if (architecture == ARCH_FOURIER_MLP) {
    return p.fail("Fourier MLP models are not supported by the decoder yet");
  }
  if (architecture != ARCH_SIREN) {
    return p.fail("unknown model architecture");
  }
  if (!(file_flags & FILE_FLAG_LINEAR_HEAD)) {
    return p.fail("SIREN with a sine on the output layer is not supported (linear head flag unset)");
  }
  if (input_dim != 3 && input_dim != 4) {
    return p.fail("input_dim must be 3 (x, y, z) or 4 (x, y, z, t)");
  }
  if (output_dim == 0 || output_dim > SRNXR_V2_MAX_OUTPUTS) {
    return p.fail("output_dim must be 1 - 8");
  }
  if (layers == 0 || layers > SRNXR_V2_MAX_LAYERS) {
    return p.fail("hidden_layer_count must be 1 - 8");
  }
  if (hidden_width == 0 || hidden_width > SRNXR_V2_MAX_WIDTH) {
    return p.fail("hidden_width must be 1 - 256");
  }
  if (network_count == 0 || leaf_count == 0 || node_count == 0) {
    return p.fail("empty model");
  }
  if (weights_dtype != DTYPE_FLOAT16 && weights_dtype != DTYPE_FLOAT32) {
    return p.fail("weights must be FP16 or FP32");
  }
  const int dims = (input_dim == 4) ? 4 : 3;
  for (int d = 0; d < dims; d++) {
    if (!std::isfinite(domain_min[d]) || !std::isfinite(domain_max[d]) ||
        !(domain_max[d] > domain_min[d]))
    {
      return p.fail("invalid domain bounds");
    }
  }

  /* Section table */
  if (section_table_offset > size || section_count > (size - section_table_offset) / section_entry_size)
  {
    return p.fail("section table extends past the end of the file");
  }
  for (uint64_t i = 0; i < section_count; ++i) {
    const uint64_t entry = section_table_offset + i * section_entry_size;
    Section section;
    section.name = std::string(data + entry, strnlen(data + entry, 32));
    section.dtype = read_value<uint32_t>(data, entry + 32);
    section.element_count = read_value<uint64_t>(data, entry + 36);
    section.byte_offset = read_value<uint64_t>(data, entry + 44);
    section.byte_length = read_value<uint64_t>(data, entry + 52);
    if (section.byte_offset > size || section.byte_length > size - section.byte_offset) {
      return p.fail("section " + section.name + " extends past the end of the file");
    }
    if (crc32(data + section.byte_offset, section.byte_length) !=
        read_value<uint32_t>(data, entry + 60))
    {
      return p.fail("section " + section.name + " checksum mismatch");
    }
    if (find_section(p.sections, section.name.c_str())) {
      return p.fail("duplicate section " + section.name);
    }
    p.sections.push_back(section);
  }

  /* Sections */
  std::string names_blob;
  std::vector<float> layer_w0, node_bounds, strict, expanded, input_tf, blend, output_tf;
  std::vector<float> weights_f, biases;
  std::vector<uint32_t> nodes, leaf_node, leaf_network, layout;
  const uint64_t L = leaf_count;

  if (!p.bytes("FIELD_NAMES", false, names_blob) || !p.floats("LAYER_W0", layers, layer_w0) ||
      !p.uints("NODES", uint64_t(node_count) * 4, nodes) ||
      !p.floats("NODE_BOUNDS", uint64_t(node_count) * 6, node_bounds) ||
      !p.uints("LEAF_NODE", L, leaf_node) ||
      !p.floats("LEAF_STRICT_BOUNDS", L * 8, strict) ||
      !p.floats("LEAF_EXPANDED_BOUNDS", L * 8, expanded) ||
      !p.floats("LEAF_INPUT_TRANSFORM", L * 8, input_tf) ||
      !p.floats("LEAF_BLEND", L * 9, blend) || !p.uints("LEAF_NETWORK", L, leaf_network) ||
      !p.floats("LEAF_OUTPUT_TRANSFORM", L * 2 * output_dim, output_tf) ||
      !p.uints("NETWORK_LAYOUT", uint64_t(network_count) * 4, layout) ||
      !p.floats("SIREN_WEIGHTS", 0, weights_f) || !p.floats("SIREN_BIASES", 0, biases))
  {
    return false;
  }
  if (!p.bytes("MODEL_METADATA_JSON", true, report.metadata_json)) {
    return false;
  }
  report.synthetic = report.metadata_json.find("\"synthetic\": true") != std::string::npos;
  report.weights_were_fp32 = (weights_dtype == DTYPE_FLOAT32);
  report.layer_w0 = layer_w0;

  /* Field names: output_dim unique NUL terminated names. */
  {
    size_t begin = 0;
    for (size_t i = 0; i < names_blob.size(); i++) {
      if (names_blob[i] == '\0') {
        report.field_names.push_back(names_blob.substr(begin, i - begin));
        begin = i + 1;
      }
    }
    if (begin != names_blob.size() || report.field_names.size() != output_dim) {
      return p.fail("FIELD_NAMES does not hold output_dim NUL terminated names");
    }
    for (size_t i = 0; i < output_dim; i++) {
      if (report.field_names[i].empty()) {
        return p.fail("empty field name");
      }
      for (size_t j = 0; j < i; j++) {
        if (report.field_names[i] == report.field_names[j]) {
          return p.fail("duplicate field name " + report.field_names[i]);
        }
      }
    }
  }

  /* Channel mapping */
  if (settings.fields.empty()) {
    if (output_dim == 1) {
      report.channel_field.push_back(0);
    }
    else {
      for (uint32_t i = 0; i < output_dim; i++) {
        if (report.field_names[i] == "rho" || report.field_names[i] == "density") {
          report.channel_field.push_back(i);
          break;
        }
      }
      if (report.channel_field.empty()) {
        return p.fail("the model has several fields, select one or three of them");
      }
    }
  }
  else {
    if (settings.fields.size() != 1 && settings.fields.size() != 3) {
      return p.fail("select one or three fields");
    }
    for (const std::string &name : settings.fields) {
      const auto it = std::find(report.field_names.begin(), report.field_names.end(), name);
      if (it == report.field_names.end()) {
        return p.fail("unknown field " + name);
      }
      report.channel_field.push_back(uint32_t(it - report.field_names.begin()));
    }
  }

  for (const float w0 : layer_w0) {
    if (!(w0 != 0.0f)) {
      return p.fail("LAYER_W0 contains zero");
    }
  }

  /* Octree topology: every node is reached exactly once from the root, internal nodes have
   * 8 children splitting them at their center, leaf nodes own exactly one leaf. */
  {
    const float tolerance = 1e-6f;
    std::vector<uint8_t> visited(node_count, 0);
    std::vector<uint32_t> stack = {0};
    if (nodes[2] != SRNXR_INVALID_INDEX || nodes[3] != 0) {
      return p.fail("node 0 must be the root");
    }
    for (int d = 0; d < 3; d++) {
      if (node_bounds[d] != domain_min[d] || node_bounds[3 + d] != domain_max[d]) {
        return p.fail("root bounds differ from the domain");
      }
    }
    std::vector<uint8_t> leaf_seen(leaf_count, 0);
    uint32_t visited_count = 0;
    while (!stack.empty()) {
      const uint32_t n = stack.back();
      stack.pop_back();
      if (visited[n]) {
        return p.fail("octree node reached twice");
      }
      visited[n] = 1;
      visited_count++;
      const uint32_t *node = &nodes[size_t(n) * 4];
      const float *b = &node_bounds[size_t(n) * 6];
      for (int d = 0; d < 3; d++) {
        if (!(b[3 + d] > b[d])) {
          return p.fail("octree node with zero extent");
        }
      }
      report.max_depth = std::max(report.max_depth, node[3]);
      if (node[3] > 32) {
        return p.fail("octree deeper than 32 levels");
      }
      if (node[0] != SRNXR_INVALID_INDEX) {
        if (node[1] != SRNXR_INVALID_INDEX) {
          return p.fail("internal octree node with a leaf");
        }
        if (uint64_t(node[0]) + 8 > node_count || node[0] == 0) {
          return p.fail("octree child index out of range");
        }
        const float center[3] = {0.5f * (b[0] + b[3]), 0.5f * (b[1] + b[4]), 0.5f * (b[2] + b[5])};
        for (uint32_t k = 0; k < 8; k++) {
          const uint32_t c = node[0] + k;
          const uint32_t *child = &nodes[size_t(c) * 4];
          const float *cb = &node_bounds[size_t(c) * 6];
          if (child[2] != n || child[3] != node[3] + 1) {
            return p.fail("octree child with a wrong parent or depth");
          }
          for (int d = 0; d < 3; d++) {
            const bool upper = (k >> d) & 1;
            const float lo = upper ? center[d] : b[d];
            const float hi = upper ? b[3 + d] : center[d];
            const float tol = tolerance * (b[3 + d] - b[d]);
            if (std::fabs(cb[d] - lo) > tol || std::fabs(cb[3 + d] - hi) > tol) {
              return p.fail("octree child bounds are not the octants of the parent");
            }
          }
          stack.push_back(c);
        }
      }
      else {
        if (node[1] >= leaf_count || leaf_seen[node[1]] || leaf_node[node[1]] != n) {
          return p.fail("octree leaf node with an invalid or duplicate leaf");
        }
        leaf_seen[node[1]] = 1;
      }
    }
    if (visited_count != node_count) {
      return p.fail("octree has unreachable nodes");
    }
  }

  /* Leaves */
  double overlap_min = std::numeric_limits<double>::max(), overlap_max = 0.0;
  for (uint32_t i = 0; i < leaf_count; i++) {
    const float *nb = &node_bounds[size_t(leaf_node[i]) * 6];
    const float *s = &strict[size_t(i) * 8];
    const float *e = &expanded[size_t(i) * 8];
    for (int d = 0; d < 3; d++) {
      if (s[d] != nb[d] || s[4 + d] != nb[3 + d]) {
        return p.fail("strict leaf bounds differ from its octree node");
      }
    }
    if (input_dim == 4 && (s[3] != domain_min[3] || s[7] != domain_max[3])) {
      return p.fail("time-partitioned leaves are not supported: the strict time range of every "
                    "leaf must be the domain time range");
    }
    for (int d = 0; d < dims; d++) {
      if (!(e[d] <= s[d]) || !(e[4 + d] >= s[4 + d])) {
        return p.fail("expanded leaf box does not contain the strict box");
      }
    }
    for (int d = 0; d < 3; d++) {
      const double extent = double(s[4 + d]) - double(s[d]);
      if (s[d] > domain_min[d]) {
        const double o = (double(s[d]) - double(e[d])) / extent;
        overlap_min = std::min(overlap_min, o);
        overlap_max = std::max(overlap_max, o);
      }
      if (s[4 + d] < domain_max[d]) {
        const double o = (double(e[4 + d]) - double(s[4 + d])) / extent;
        overlap_min = std::min(overlap_min, o);
        overlap_max = std::max(overlap_max, o);
      }
    }
    const float *it = &input_tf[size_t(i) * 8];
    for (uint32_t d = 0; d < input_dim; d++) {
      if (!(it[4 + d] != 0.0f)) {
        return p.fail("leaf input scale is zero");
      }
    }
    const float *bl = &blend[size_t(i) * 9];
    if (!(bl[8] > 0.0f)) {
      return p.fail("leaf blend sigma must be positive");
    }
    for (int d = 0; d < 4; d++) {
      if (!(bl[4 + d] >= 0.0f)) {
        return p.fail("leaf blend axis scale must be non-negative");
      }
    }
    if (leaf_network[i] >= network_count) {
      return p.fail("leaf references a missing network");
    }
  }
  report.overlap_min = (overlap_min == std::numeric_limits<double>::max()) ? 0.0 : overlap_min;
  report.overlap_max = overlap_max;

  /* Networks */
  for (uint32_t n = 0; n < network_count; n++) {
    const uint64_t width = layout[n * 4 + 0];
    const uint64_t weight_offset = layout[n * 4 + 1];
    const uint64_t bias_offset = layout[n * 4 + 2];
    const uint64_t weight_count = width * input_dim + (layers - 1) * width * width +
                                  output_dim * width;
    const uint64_t bias_count = uint64_t(layers) * width + output_dim;
    if (width == 0 || width > hidden_width) {
      return p.fail("network width is zero or exceeds hidden_width");
    }
    if (weight_offset + weight_count > weights_f.size() || bias_offset + bias_count > biases.size())
    {
      return p.fail("network weights or biases out of range");
    }
  }
  if (weights_f.size() >= uint64_t(SRNXR_INVALID_INDEX) || biases.size() >= uint64_t(SRNXR_INVALID_INDEX)) {
    return p.fail("weight arrays exceed 32-bit element offsets");
  }

  /* Deployed weights: FP16, round to nearest even when the file stores FP32. */
  std::vector<uint16_t> weights_h(weights_f.size());
  for (size_t i = 0; i < weights_f.size(); i++) {
    weights_h[i] = float_to_half_bits(weights_f[i]);
    weights_f[i] = half_bits_to_float(weights_h[i]);
    if (!std::isfinite(weights_f[i])) {
      return p.fail("weight overflows FP16");
    }
  }
  report.parse_seconds = seconds_since(parse_start);

  const auto pack_start = std::chrono::steady_clock::now();

  /* Device nodes */
  std::vector<SRNXRNodeV2> dnodes(node_count);
  for (uint32_t n = 0; n < node_count; n++) {
    SRNXRNodeV2 &dn = dnodes[n];
    for (int d = 0; d < 3; d++) {
      dn.bmin[d] = node_bounds[size_t(n) * 6 + d];
      dn.bmax[d] = node_bounds[size_t(n) * 6 + 3 + d];
    }
    dn.first_child = nodes[size_t(n) * 4 + 0];
    dn.leaf = nodes[size_t(n) * 4 + 1];
  }

  /* Neighbor lists: leaf i lists every leaf j whose expanded box intersects the strict box of
   * i, so every network whose support contains a point of leaf i is in the list. Built by
   * querying the octree with the expanded box of every leaf. */
  std::vector<std::vector<uint32_t>> neighbor_lists(leaf_count);
  for (uint32_t j = 0; j < leaf_count; j++) {
    const float *e = &expanded[size_t(j) * 8];
    const float emax[3] = {e[4], e[5], e[6]};
    std::vector<uint32_t> stack = {0};
    while (!stack.empty()) {
      const SRNXRNodeV2 &dn = dnodes[stack.back()];
      stack.pop_back();
      if (!box_intersects(dn.bmin, dn.bmax, e, emax, 3)) {
        continue;
      }
      if (dn.first_child == SRNXR_INVALID_INDEX) {
        const float *s = &strict[size_t(dn.leaf) * 8];
        if (dims == 3 || box_intersects(s, s + 4, e, e + 4, 4)) {
          neighbor_lists[dn.leaf].push_back(j);
        }
      }
      else {
        for (uint32_t k = 0; k < 8; k++) {
          stack.push_back(dn.first_child + k);
        }
      }
    }
  }
  std::vector<uint32_t> neighbors;
  uint64_t neighbor_total = 0;
  for (uint32_t i = 0; i < leaf_count; i++) {
    std::sort(neighbor_lists[i].begin(), neighbor_lists[i].end());
    if (!std::binary_search(neighbor_lists[i].begin(), neighbor_lists[i].end(), i)) {
      return p.fail("internal error: a leaf is not in its own neighbor list");
    }
    report.max_neighbors = std::max(report.max_neighbors, uint32_t(neighbor_lists[i].size()));
    neighbor_total += neighbor_lists[i].size();
  }
  report.mean_neighbors = double(neighbor_total) / leaf_count;

  /* Certified output range of every leaf: |sin| <= 1, so the linear head output y_k is within
   * b_k +- sum_j |W_kj|. Evaluated in double from the deployed FP16 weights and widened by a
   * relative margin that covers the FP32 rounding of the decoders. */
  std::vector<double> leaf_lo(size_t(leaf_count) * output_dim), leaf_hi(leaf_lo.size());
  for (uint32_t i = 0; i < leaf_count; i++) {
    const uint32_t n = leaf_network[i];
    const uint64_t width = layout[n * 4 + 0];
    const uint64_t head_offset = layout[n * 4 + 1] + width * input_dim +
                                 (layers - 1) * width * width;
    const uint64_t head_bias = layout[n * 4 + 2] + uint64_t(layers) * width;
    for (uint32_t k = 0; k < output_dim; k++) {
      double radius = 0.0;
      for (uint64_t j = 0; j < width; j++) {
        radius += std::fabs(double(weights_f[head_offset + k * width + j]));
      }
      const double b = biases[head_bias + k];
      const double margin = 1e-4 * (std::fabs(b) + radius) + 1e-6;
      const double scale = output_tf[size_t(i) * 2 * output_dim + k];
      const double offset = output_tf[size_t(i) * 2 * output_dim + output_dim + k];
      const double a = (b - radius - margin) * scale + offset;
      const double c = (b + radius + margin) * scale + offset;
      leaf_lo[size_t(i) * output_dim + k] = std::min(a, c);
      leaf_hi[size_t(i) * output_dim + k] = std::max(a, c);
    }
  }

  /* Device leaves */
  std::vector<SRNXRLeafV2> dleaves(leaf_count);
  for (uint32_t i = 0; i < leaf_count; i++) {
    SRNXRLeafV2 &dl = dleaves[i];
    memset(&dl, 0, sizeof(dl));
    for (int d = 0; d < 4; d++) {
      dl.expanded_min[d] = expanded[size_t(i) * 8 + d];
      dl.expanded_max[d] = expanded[size_t(i) * 8 + 4 + d];
      dl.input_offset[d] = input_tf[size_t(i) * 8 + d];
      dl.input_scale[d] = input_tf[size_t(i) * 8 + 4 + d];
      dl.blend_center[d] = blend[size_t(i) * 9 + d];
      dl.blend_scale[d] = blend[size_t(i) * 9 + 4 + d];
    }
    if (dims == 3) {
      /* Time does not take part in 3D models. */
      dl.expanded_min[3] = -std::numeric_limits<float>::max();
      dl.expanded_max[3] = std::numeric_limits<float>::max();
      dl.blend_scale[3] = 0.0f;
    }
    const float sigma = blend[size_t(i) * 9 + 8];
    dl.blend_inv_two_sigma_sq = 0.5f / (sigma * sigma);
    dl.network = leaf_network[i];
    dl.neighbor_begin = uint32_t(neighbors.size());
    neighbors.insert(neighbors.end(), neighbor_lists[i].begin(), neighbor_lists[i].end());
    dl.neighbor_end = uint32_t(neighbors.size());
    for (uint32_t k = 0; k < output_dim; k++) {
      dl.output_scale[k] = output_tf[size_t(i) * 2 * output_dim + k];
      dl.output_offset[k] = output_tf[size_t(i) * 2 * output_dim + output_dim + k];
    }
    /* Range of the output of this leaf network; the bake takes the union over the leaves whose
     * expanded box touches the voxel, of which the blended value is a convex combination. */
    for (size_t c = 0; c < report.channel_field.size(); c++) {
      const double lo = leaf_lo[size_t(i) * output_dim + report.channel_field[c]];
      const double hi = leaf_hi[size_t(i) * output_dim + report.channel_field[c]];
      dl.bound_min[c] = float(lo);
      dl.bound_max[c] = float(hi);
      dl.bound_min[c] = std::nextafter(dl.bound_min[c], -std::numeric_limits<float>::max());
      dl.bound_max[c] = std::nextafter(dl.bound_max[c], std::numeric_limits<float>::max());
    }
  }

  /* Networks, with the mma.m16n8k16 B fragment copy of the hidden and output layers: for an
   * output tile nt (8 rows of W) and an input tile kt (16 columns of W), lane l = 4 g + t holds
   * {W[8 nt + g][16 kt + 2 t], W[..][.. + 1]} and {W[8 nt + g][16 kt + 2 t + 8], W[..][.. + 9]}
   * as FP16 pairs, the lower column in the lower half. Output rows beyond output_dim are zero. */
  const bool want_mma = settings.backend == SRNXR_BACKEND_CUDA_TENSOR || settings.force_mma_weights;
  std::vector<SRNXRNetworkV2> dnetworks(network_count);
  std::vector<uint32_t> mma;
  /* Row-major FP16 copy for the SIMT path, every network starts at a multiple of 8 elements so
   * rows of widths divisible by 8 are 16-byte aligned. */
  std::vector<uint16_t> dweights;
  /* CUDA_TENSOR needs it too, for the warps that are not fully active. */
  const bool want_warp = ((settings.backend == SRNXR_BACKEND_CUDA_FUSED ||
                           settings.backend == SRNXR_BACKEND_CUDA_TENSOR) &&
                          settings.warp_weights) ||
                         settings.force_warp_weights;
  std::vector<uint16_t> warp;
  for (uint32_t n = 0; n < network_count; n++) {
    SRNXRNetworkV2 &dn = dnetworks[n];
    dn.width = layout[n * 4 + 0];
    dn.bias_offset = layout[n * 4 + 2];
    dn.mma_offset = SRNXR_INVALID_INDEX;
    dn.coopvec_offset = SRNXR_INVALID_INDEX;
    const uint32_t w = dn.width;
    const uint64_t file_offset = layout[n * 4 + 1];
    const uint64_t weight_count = uint64_t(w) * input_dim + uint64_t(layers - 1) * w * w +
                                  uint64_t(output_dim) * w;
    dweights.resize((dweights.size() + 7) / 8 * 8, 0);
    dn.weight_offset = uint32_t(dweights.size());
    dweights.insert(dweights.end(),
                    weights_h.begin() + file_offset,
                    weights_h.begin() + file_offset + weight_count);
    if (dweights.size() >= uint64_t(SRNXR_INVALID_INDEX)) {
      return p.fail("weight array exceeds 32-bit element offsets");
    }
    /* Transposed hidden layers [in][out] for the warp-cooperative SIMT path: lane l reads the
     * outputs 2 l + 64 j, 2 l + 64 j + 1 of input i as one 32-bit word. */
    dn.warp_offset = SRNXR_INVALID_INDEX;
    if (want_warp && (w % 64) == 0) {
      dn.warp_offset = uint32_t(warp.size());
      for (uint32_t l = 1; l < layers; l++) {
        const uint64_t layer_offset = file_offset + uint64_t(w) * input_dim +
                                      uint64_t(l - 1) * w * w;
        for (uint32_t i = 0; i < w; i++) {
          for (uint32_t o = 0; o < w; o++) {
            warp.push_back(weights_h[layer_offset + uint64_t(o) * w + i]);
          }
        }
      }
      if (warp.size() >= uint64_t(SRNXR_INVALID_INDEX)) {
        return p.fail("warp weight copy exceeds 32-bit offsets");
      }
    }
    else if (want_warp) {
      report.networks_without_warp++;
    }
    if (!want_mma) {
      continue;
    }
    if (!(w == 64 || w == 128 || w == 256)) {
      report.networks_without_mma++;
      continue;
    }
    dn.mma_offset = uint32_t(mma.size());
    uint64_t offset = file_offset + uint64_t(w) * input_dim;
    auto pack_layer = [&](const uint32_t rows, const uint32_t row_tiles) {
      for (uint32_t nt = 0; nt < row_tiles; nt++) {
        for (uint32_t kt = 0; kt < w / 16; kt++) {
          for (uint32_t lane = 0; lane < 32; lane++) {
            const uint32_t g = lane >> 2, t = lane & 3;
            const uint32_t row = nt * 8 + g;
            uint16_t v[4] = {0, 0, 0, 0};
            if (row < rows) {
              const uint64_t r = offset + uint64_t(row) * w + kt * 16 + 2 * t;
              v[0] = weights_h[r + 0];
              v[1] = weights_h[r + 1];
              v[2] = weights_h[r + 8];
              v[3] = weights_h[r + 9];
            }
            mma.push_back(uint32_t(v[0]) | (uint32_t(v[1]) << 16));
            mma.push_back(uint32_t(v[2]) | (uint32_t(v[3]) << 16));
          }
        }
      }
      offset += uint64_t(rows) * w;
    };
    for (uint32_t l = 1; l < layers; l++) {
      pack_layer(w, w / 8);
    }
    pack_layer(output_dim, 1);
    if (mma.size() >= uint64_t(SRNXR_INVALID_INDEX)) {
      return p.fail("mma weight copy exceeds 32-bit offsets");
    }
  }
  if (want_mma && report.networks_without_mma) {
    char buf[256];
    snprintf(buf,
             sizeof(buf),
             "%u networks have a width without a Tensor Core specialization (64, 128, 256), "
             "CUDA_TENSOR evaluates them with the SIMT path and counts them as fallbacks",
             report.networks_without_mma);
    report.warnings.push_back(buf);
  }

  /* Cooperative vector region, the matrices are converted on the device by the caller. */
  std::vector<uint8_t> coop;
  if (settings.backend == SRNXR_BACKEND_COOPVEC) {
    auto align64 = [](const uint64_t v) { return (v + 63) / 64 * 64; };
    /* Without an OptiX context the sizes are assumed to equal the row-major FP16 size rounded to
     * 64 bytes, as optixCoopVecMatrixComputeSize() reports on SM 12.0 with OptiX 9.1;
     * OptiXDevice verifies them before converting and refuses the volume otherwise. */
    const auto matrix_size = settings.coopvec_matrix_size ?
                                 settings.coopvec_matrix_size :
                                 [](uint32_t rows, uint32_t k) -> uint64_t {
                                   return (uint64_t(rows) * k * 2 + 63) / 64 * 64;
                                 };
    for (uint32_t n = 0; n < network_count; n++) {
      SRNXRNetworkV2 &dn = dnetworks[n];
      dn.coopvec_offset = SRNXR_INVALID_INDEX;
      const uint32_t w = dn.width;
      if (!(w == 64 || w == 128 || w == 256)) {
        continue;
      }
      uint64_t offset = align64(coop.size());
      dn.coopvec_offset = uint32_t(offset);
      const uint64_t file_offset = layout[n * 4 + 1];
      uint64_t src = file_offset + uint64_t(w) * input_dim;
      auto add_matrix = [&](const uint32_t rows, const uint32_t real_rows) {
        SRNXRCoopVecMatrix m;
        m.n = rows;
        m.k = w;
        m.size = matrix_size(rows, w);
        m.blob_offset = offset;
        report.coopvec_staging.resize(align64(report.coopvec_staging.size() * 2) / 2, 0);
        m.staging_offset = report.coopvec_staging.size() * 2;
        for (uint32_t r = 0; r < rows; r++) {
          for (uint32_t c = 0; c < w; c++) {
            report.coopvec_staging.push_back((r < real_rows) ? weights_h[src + uint64_t(r) * w + c] :
                                                               uint16_t(0));
          }
        }
        src += uint64_t(real_rows) * w;
        offset = align64(offset + m.size);
        report.coopvec_matrices.push_back(m);
      };
      for (uint32_t l = 1; l < layers; l++) {
        add_matrix(w, w);
      }
      add_matrix(8, output_dim);
      /* Biases of the hidden layers and of the output layer padded to 8, FP32 then FP16. */
      std::vector<float> b32;
      const uint64_t bias = layout[n * 4 + 2];
      for (uint32_t l = 1; l < layers; l++) {
        for (uint32_t i = 0; i < w; i++) {
          b32.push_back(biases[bias + uint64_t(l) * w + i]);
        }
      }
      for (uint32_t k = 0; k < 8; k++) {
        b32.push_back((k < output_dim) ? biases[bias + uint64_t(layers) * w + k] : 0.0f);
      }
      coop.resize(offset + align64(b32.size() * 4) + align64(b32.size() * 2), 0);
      memcpy(coop.data() + offset, b32.data(), b32.size() * 4);
      offset += align64(b32.size() * 4);
      for (size_t i = 0; i < b32.size(); i++) {
        const uint16_t h = float_to_half_bits(b32[i]);
        memcpy(coop.data() + offset + i * 2, &h, 2);
      }
      if (coop.size() >= uint64_t(SRNXR_INVALID_INDEX)) {
        return p.fail("cooperative vector region exceeds 32-bit offsets");
      }
    }
  }

  /* Header */
  SRNXRDeviceHeaderV2 hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.magic = SRNXR_DEVICE_MAGIC;
  hdr.version = SRNXR_DEVICE_VERSION_V2;
  hdr.input_dim = input_dim;
  hdr.output_dim = output_dim;
  hdr.hidden_layer_count = layers;
  hdr.hidden_width = hidden_width;
  hdr.network_count = network_count;
  hdr.leaf_count = leaf_count;
  hdr.node_count = node_count;
  hdr.flags = (settings.counters ? SRNXR_DEVICE_FLAG_COUNTERS : 0) |
              (settings.partial_warps ? SRNXR_DEVICE_FLAG_PARTIAL_WARPS : 0) |
              (settings.sampled_majorant ? SRNXR_DEVICE_FLAG_SAMPLED_MAJORANT : 0) |
              ((want_mma && report.networks_without_mma < network_count) ?
                   SRNXR_DEVICE_FLAG_MMA_WEIGHTS :
                   0) |
              ((want_warp && report.networks_without_warp < network_count) ?
                   SRNXR_DEVICE_FLAG_WARP_WEIGHTS :
                   0);
  if (settings.sampled_majorant) {
    report.warnings.push_back("sampled volume majorant: the bake samples the decoded field, the "
                              "majorant is not certified");
  }
  hdr.backend = settings.backend;
  hdr.sine_mode = settings.sine_mode;
  hdr.channel_count = uint32_t(report.channel_field.size());
  for (size_t c = 0; c < report.channel_field.size(); c++) {
    hdr.channel_field[c] = report.channel_field[c];
  }
  for (int d = 0; d < 4; d++) {
    hdr.domain_min[d] = domain_min[d];
    hdr.domain_max[d] = domain_max[d];
  }
  if (dims == 3) {
    hdr.domain_min[3] = -std::numeric_limits<float>::max();
    hdr.domain_max[3] = std::numeric_limits<float>::max();
  }
  if (dims == 4) {
    if (settings.has_time_frame) {
      hdr.time_frame = settings.time_frame;
      if (!(settings.time_frame >= domain_min[3] && settings.time_frame <= domain_max[3])) {
        report.warnings.push_back("simulation time is outside the model time range, every "
                                  "sample is outside the domain");
      }
    }
    else {
      hdr.time_frame = domain_min[3];
      report.warnings.push_back("no simulation time given, using the start of the model time "
                                "range");
    }
  }
  hdr.time_shutter = settings.time_shutter;
  for (int d = 0; d < 3; d++) {
    hdr.bake_margin[d] = 2.0f * (domain_max[d] - domain_min[d]) /
                         float(1 << std::max(0, std::min(settings.bake_grid_depth, 16)));
  }
  hdr.outside_value = settings.outside_value;
  hdr.coopvec_mode = settings.coopvec_mode;
  {
    /* FNV-1a over everything the decoded values depend on except the channel mapping. */
    uint32_t outside_bits;
    std::memcpy(&outside_bits, &hdr.outside_value, sizeof(outside_bits));
    const uint32_t parts[] = {read_value<uint32_t>(data, file_crc_offset),
                              uint32_t(size),
                              uint32_t(uint64_t(size) >> 32),
                              hdr.flags,
                              hdr.backend,
                              hdr.sine_mode,
                              hdr.coopvec_mode,
                              outside_bits};
    uint32_t key = 2166136261u;
    for (const uint32_t part : parts) {
      for (int b = 0; b < 4; b++) {
        key = (key ^ ((part >> (8 * b)) & 0xFFu)) * 16777619u;
      }
    }
    hdr.decode_key = (key == 0) ? 1 : key;
  }
  for (uint32_t l = 0; l < layers; l++) {
    hdr.layer_w0[l] = layer_w0[l];
  }

  blob.resize((sizeof(SRNXRDeviceHeaderV2) + SRNXR_DEVICE_ALIGNMENT - 1) / SRNXR_DEVICE_ALIGNMENT *
                  SRNXR_DEVICE_ALIGNMENT,
              0);
  hdr.nodes_offset = append_array(blob, dnodes.data(), dnodes.size());
  hdr.leaves_offset = append_array(blob, dleaves.data(), dleaves.size());
  hdr.neighbors_offset = append_array(blob, neighbors.data(), neighbors.size());
  hdr.networks_offset = append_array(blob, dnetworks.data(), dnetworks.size());
  hdr.weights_offset = append_array(blob, dweights.data(), dweights.size());
  hdr.biases_offset = append_array(blob, biases.data(), biases.size());
  hdr.mma_weights_offset = append_array(blob, mma.data(), mma.size());
  hdr.warp_weights_offset = append_array(blob, warp.data(), warp.size());
  if (!coop.empty()) {
    hdr.coopvec_offset = append_array(blob, coop.data(), coop.size());
    hdr.coopvec_size = coop.size();
    hdr.coopvec_staging_offset = append_array(
        blob, report.coopvec_staging.data(), report.coopvec_staging.size());
    std::vector<SRNXRCoopVecMatrixV2> table;
    for (SRNXRCoopVecMatrix &m : report.coopvec_matrices) {
      m.blob_offset += hdr.coopvec_offset;
      table.push_back(
          {m.n, m.k, hdr.coopvec_staging_offset + m.staging_offset, m.blob_offset, m.size});
    }
    hdr.coopvec_table_offset = append_array(blob, table.data(), table.size());
    hdr.coopvec_matrix_count = table.size();
    report.coopvec_bytes = coop.size();
  }
  const uint64_t counters[SRNXR_COUNTER_NUM] = {0};
  hdr.counters_offset = append_array(blob, counters, SRNXR_COUNTER_NUM);
  blob.resize((blob.size() + SRNXR_DEVICE_ALIGNMENT - 1) / SRNXR_DEVICE_ALIGNMENT *
                  SRNXR_DEVICE_ALIGNMENT,
              0);
  hdr.total_size = blob.size();
  memcpy(blob.data(), &hdr, sizeof(hdr));

  report.device_bytes = blob.size();
  report.weight_bytes = dweights.size() * sizeof(uint16_t) + biases.size() * sizeof(float);
  report.mma_weight_bytes = mma.size() * sizeof(uint32_t);
  report.warp_weight_bytes = warp.size() * sizeof(uint16_t);
  report.pack_seconds = seconds_since(pack_start);
  return true;
}

CCL_NAMESPACE_END
