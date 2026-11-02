/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone OptiX cooperative vector (COOPVEC) capability probe, validation and benchmark of
 * the SRNXR version 2 decoder.
 *
 *   srnxr_coopvec model.srnxr points.f32 [options]
 *     --probe                    only report the device capability and matrix sizes
 *     --mode fp16|fp32           matrix product outputs and biases in FP16 (default) or FP32;
 *                                fp32 needs the programs built with -DSRNXR_COOPVEC_TRY_FP32_OUTPUT
 *                                and OptiX 9.1 rejects them
 *     --sine accurate|fast       sinf or __sinf for the per-element sine
 *     --out values.f32           decoded fields, float32 [N, output_dim]
 *     --batches 1,32,...         timed launch sizes, --repeat R timed launches each
 *     --json results.json
 *     --ptx file.ptx             OptiX programs (default: next to the executable)
 *
 * Exit code 2 when the device or the driver has no cooperative vector support: COOPVEC is
 * never replaced by another backend. */

#include "srnxr_compat.h"

#include "scene/srnxr_model.h"
#include "srnxr_coopvec_params.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ccl;

#define CUDA_CHECK(x) \
  do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
              cudaGetErrorString(err_)); \
      exit(1); \
    } \
  } while (0)

#define OPTIX_CHECK(x) \
  do { \
    OptixResult res_ = (x); \
    if (res_ != OPTIX_SUCCESS) { \
      fprintf(stderr, "OptiX error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
              optixGetErrorString(res_)); \
      exit(1); \
    } \
  } while (0)

template<typename T> struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};
struct EmptyData {
  int unused;
};

static bool read_file(const std::string &path, std::vector<char> &data)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return false;
  }
  data.resize(size_t(f.tellg()));
  f.seekg(0);
  return bool(f.read(data.data(), data.size()));
}

static std::string json_escape(const std::string &s)
{
  std::string r;
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      r += '\\';
    }
    if (c == '\n') {
      r += "\\n";
      continue;
    }
    r += c;
  }
  return r;
}

static void log_cb(unsigned int level, const char *tag, const char *message, void *)
{
  if (level <= 3) {
    fprintf(stderr, "[OptiX %u][%s] %s\n", level, tag, message);
  }
}

int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.srnxr points.f32 [options]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1], points_path = argv[2];
  std::string mode = "fp16", sine = "accurate", out_path, json_path;
  std::string ptx_path = std::string(argv[0]) + "_device.ptx";
  std::vector<int> batches;
  int repeat = 20;
  bool probe = false, counters = true;
  for (int i = 3; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (a == "--probe") {
      probe = true;
    }
    else if (a == "--mode") {
      mode = next();
    }
    else if (a == "--sine") {
      sine = next();
    }
    else if (a == "--out") {
      out_path = next();
    }
    else if (a == "--json") {
      json_path = next();
    }
    else if (a == "--ptx") {
      ptx_path = next();
    }
    else if (a == "--repeat") {
      repeat = atoi(next().c_str());
    }
    else if (a == "--no-counters") {
      counters = false;
    }
    else if (a == "--batches") {
      std::stringstream ss(next());
      std::string item;
      while (std::getline(ss, item, ',')) {
        batches.push_back(atoi(item.c_str()));
      }
    }
    else {
      fprintf(stderr, "unknown option %s\n", a.c_str());
      return 1;
    }
  }

  std::string json = "{\n";
  auto jkv = [&](const char *key, const std::string &value, bool quote = true) {
    json += std::string("  \"") + key + "\": " +
            (quote ? "\"" + json_escape(value) + "\"" : value) + ",\n";
  };
  auto finish = [&](int code) {
    json += "  \"exit_code\": " + std::to_string(code) + "\n}\n";
    if (!json_path.empty()) {
      std::ofstream f(json_path);
      f << json;
    }
    return code;
  };

  /* Device and OptiX context */
  CUDA_CHECK(cudaFree(nullptr));
  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int driver = 0;
  cudaDriverGetVersion(&driver);
  printf("GPU %s, SM %d.%d, CUDA driver %d, OptiX SDK headers %d\n", prop.name, prop.major,
         prop.minor, driver, OPTIX_VERSION);
  jkv("gpu", prop.name);
  jkv("sm", std::to_string(prop.major * 10 + prop.minor), false);
  jkv("cuda_driver_version", std::to_string(driver), false);
  jkv("optix_sdk_version", std::to_string(OPTIX_VERSION), false);
  jkv("backend_requested", "COOPVEC");
  jkv("coopvec_mode", mode);
  jkv("sine_mode", sine);

  OptixResult init = optixInit();
  if (init != OPTIX_SUCCESS) {
    fprintf(stderr, "optixInit failed: %s (driver without OptiX %d ABI?)\n",
            optixGetErrorString(init), OPTIX_VERSION);
    jkv("error", std::string("optixInit: ") + optixGetErrorString(init));
    return finish(2);
  }
  OptixDeviceContextOptions ctx_options = {};
  ctx_options.logCallbackFunction = log_cb;
  ctx_options.logCallbackLevel = 3;
  OptixDeviceContext ctx = nullptr;
  OPTIX_CHECK(optixDeviceContextCreate(nullptr, &ctx_options, &ctx));

  unsigned int coop_flags = 0;
  OPTIX_CHECK(optixDeviceContextGetProperty(ctx, OPTIX_DEVICE_PROPERTY_COOP_VEC, &coop_flags,
                                            sizeof(coop_flags)));
  printf("OPTIX_DEVICE_PROPERTY_COOP_VEC = 0x%x (%s)\n", coop_flags,
         (coop_flags & OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD) ? "STANDARD" : "NONE");
  jkv("optix_coop_vec_property", std::to_string(coop_flags), false);
  if (!(coop_flags & OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD)) {
    fprintf(stderr, "The device has no cooperative vector support, COOPVEC is unavailable\n");
    jkv("error", "OPTIX_DEVICE_PROPERTY_COOP_VEC has no STANDARD flag");
    return finish(2);
  }
  for (const uint32_t w : {64u, 128u, 256u}) {
    size_t s_ww = 0, s_head = 0, s_row = 0;
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(ctx, w, w, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                              OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, 0,
                                              &s_ww));
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(ctx, 8, w, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                              OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, 0,
                                              &s_head));
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(ctx, w, w, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                              OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR, 0, &s_row));
    printf("  FP16 %ux%u: INFERENCING_OPTIMAL %zu bytes (row-major %zu), 8x%u: %zu bytes\n", w,
           w, s_ww, s_row, w, s_head);
    jkv(("matrix_bytes_" + std::to_string(w) + "x" + std::to_string(w)).c_str(),
        std::to_string(s_ww), false);
  }
  if (probe) {
    optixDeviceContextDestroy(ctx);
    return finish(0);
  }

  /* Model */
  std::vector<char> file, pts_raw;
  if (!read_file(model_path, file)) {
    fprintf(stderr, "cannot read %s\n", model_path.c_str());
    return 1;
  }
  if (!read_file(points_path, pts_raw) || pts_raw.size() % 16) {
    fprintf(stderr, "cannot read %s\n", points_path.c_str());
    return 1;
  }
  const int n = int(pts_raw.size() / 16);
  SRNXRBuildSettings settings;
  settings.backend = SRNXR_BACKEND_COOPVEC;
  settings.coopvec_mode = (mode == "fp16") ? 1 : 0;
  settings.counters = counters;
  if (!srnxr_sine_mode_from_name(sine, settings.sine_mode)) {
    fprintf(stderr, "unknown sine mode %s\n", sine.c_str());
    return 1;
  }
  settings.has_time_frame = true;
  settings.coopvec_matrix_size = [&](uint32_t rows, uint32_t k) -> uint64_t {
    size_t s = 0;
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(ctx, rows, k, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                                              OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, 0,
                                              &s));
    return s;
  };
  std::vector<char> blob;
  SRNXRBuildReport report;
  std::string error;
  if (!srnxr_build_device_blob_v2(file.data(), file.size(), settings, blob, report, error)) {
    fprintf(stderr, "SRNXR load failed: %s\n", error.c_str());
    return 1;
  }
  SRNXRDeviceHeaderV2 hdr;
  memcpy(&hdr, blob.data(), sizeof(hdr));
  printf("SRNXR model %s\n%s  Cooperative vector region: %zu bytes, %zu matrices\n",
         model_path.c_str(), report.summary().c_str(), report.coopvec_bytes,
         report.coopvec_matrices.size());
  jkv("synthetic_model", report.synthetic ? "true" : "false", false);
  jkv("load_report", report.summary());
  jkv("coopvec_region_bytes", std::to_string(report.coopvec_bytes), false);
  if (report.coopvec_matrices.empty()) {
    fprintf(stderr, "no network has a width with a cooperative vector specialization\n");
    return finish(2);
  }

  uint8_t *d_blob = nullptr, *d_staging = nullptr;
  CUDA_CHECK(cudaMalloc(&d_blob, blob.size()));
  CUDA_CHECK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_staging, report.coopvec_staging.size() * 2));
  CUDA_CHECK(cudaMemcpy(d_staging, report.coopvec_staging.data(),
                        report.coopvec_staging.size() * 2, cudaMemcpyHostToDevice));

  /* Matrix layout conversion on the device */
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto convert_start = std::chrono::steady_clock::now();
  for (const SRNXRCoopVecMatrix &m : report.coopvec_matrices) {
    OptixCoopVecMatrixDescription in_desc = {};
    in_desc.N = m.n;
    in_desc.K = m.k;
    in_desc.offsetInBytes = 0;
    in_desc.elementType = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
    in_desc.layout = OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
    in_desc.rowColumnStrideInBytes = m.k * 2;
    in_desc.sizeInBytes = m.n * m.k * 2;
    OptixCoopVecMatrixDescription out_desc = in_desc;
    out_desc.layout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
    out_desc.rowColumnStrideInBytes = 0;
    out_desc.sizeInBytes = uint32_t(m.size);
    OptixNetworkDescription in_net = {&in_desc, 1};
    OptixNetworkDescription out_net = {&out_desc, 1};
    OPTIX_CHECK(optixCoopVecMatrixConvert(ctx, nullptr, 1, &in_net,
                                          CUdeviceptr(d_staging + m.staging_offset), 0, &out_net,
                                          CUdeviceptr(d_blob + m.blob_offset), 0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const double convert_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                         convert_start).count();
  printf("optixCoopVecMatrixConvert of %zu matrices: %.3f ms\n", report.coopvec_matrices.size(),
         convert_s * 1e3);
  jkv("matrix_convert_ms", std::to_string(convert_s * 1e3), false);
  CUDA_CHECK(cudaFree(d_staging));

  /* Pipeline */
  std::vector<char> ptx;
  if (!read_file(ptx_path, ptx)) {
    fprintf(stderr, "cannot read %s\n", ptx_path.c_str());
    return 1;
  }
  OptixModuleCompileOptions module_options = {};
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
  OptixPipelineCompileOptions pipeline_options = {};
  pipeline_options.usesMotionBlur = 0;
  pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipeline_options.numPayloadValues = 0;
  pipeline_options.numAttributeValues = 0;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
  char log[8192];
  size_t log_size = sizeof(log);
  OptixModule module = nullptr;
  const auto compile_start = std::chrono::steady_clock::now();
  OPTIX_CHECK(optixModuleCreate(ctx, &module_options, &pipeline_options, ptx.data(), ptx.size(),
                                log, &log_size, &module));
  OptixProgramGroupOptions pg_options = {};
  OptixProgramGroupDesc rg_desc = {};
  rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  rg_desc.raygen.module = module;
  rg_desc.raygen.entryFunctionName = "__raygen__srnxr";
  OptixProgramGroupDesc ms_desc = {};
  ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  ms_desc.miss.module = module;
  ms_desc.miss.entryFunctionName = "__miss__srnxr";
  OptixProgramGroup groups[2];
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(ctx, &rg_desc, 1, &pg_options, log, &log_size, &groups[0]));
  log_size = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(ctx, &ms_desc, 1, &pg_options, log, &log_size, &groups[1]));
  OptixPipelineLinkOptions link_options = {};
  link_options.maxTraceDepth = 1;
  OptixPipeline pipeline = nullptr;
  log_size = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(ctx, &pipeline_options, &link_options, groups, 2, log,
                                  &log_size, &pipeline));
  OptixStackSizes stack_sizes = {};
  for (OptixProgramGroup g : groups) {
    OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stack_sizes, pipeline));
  }
  printf("raygen stack: CSS %u bytes\n", stack_sizes.cssRG);
  jkv("raygen_continuation_stack_bytes", std::to_string(stack_sizes.cssRG), false);
  uint32_t dc_trav = 0, dc_state = 0, continuation = 0;
  OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1, 0, 0, &dc_trav, &dc_state,
                                         &continuation));
  OPTIX_CHECK(optixPipelineSetStackSize(pipeline, dc_trav, dc_state, continuation, 1));
  const double compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                         compile_start).count();
  jkv("pipeline_compile_ms", std::to_string(compile_s * 1e3), false);

  SbtRecord<EmptyData> rg_rec = {}, ms_rec = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(groups[0], &rg_rec));
  OPTIX_CHECK(optixSbtRecordPackHeader(groups[1], &ms_rec));
  void *d_rg = nullptr, *d_ms = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rg, sizeof(rg_rec)));
  CUDA_CHECK(cudaMalloc(&d_ms, sizeof(ms_rec)));
  CUDA_CHECK(cudaMemcpy(d_rg, &rg_rec, sizeof(rg_rec), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ms, &ms_rec, sizeof(ms_rec), cudaMemcpyHostToDevice));
  OptixShaderBindingTable sbt = {};
  sbt.raygenRecord = CUdeviceptr(d_rg);
  sbt.missRecordBase = CUdeviceptr(d_ms);
  sbt.missRecordStrideInBytes = sizeof(ms_rec);
  sbt.missRecordCount = 1;

  /* Data */
  const int output_dim = int(hdr.output_dim);
  float4 *d_points = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_points, pts_raw.size()));
  CUDA_CHECK(cudaMemcpy(d_points, pts_raw.data(), pts_raw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_out, size_t(n) * output_dim * sizeof(float)));
  SRNXRCoopVecParams params = {d_blob, d_points, d_out, n, output_dim};
  SRNXRCoopVecParams *d_params = nullptr;
  CUDA_CHECK(cudaMalloc(&d_params, sizeof(params)));

  auto launch = [&](int count) {
    params.n = count;
    CUDA_CHECK(cudaMemcpy(d_params, &params, sizeof(params), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pipeline, nullptr, CUdeviceptr(d_params), sizeof(params), &sbt,
                            unsigned(count), 1, 1));
  };
  auto read_counters = [&]() {
    uint64_t c[SRNXR_COUNTER_NUM];
    CUDA_CHECK(cudaMemcpy(c, d_blob + hdr.counters_offset, sizeof(c), cudaMemcpyDeviceToHost));
    std::string js = "{";
    for (int k = 0; k <= SRNXR_COUNTER_WARP_ACTIVE_LANES; k++) {
      js += std::string(k ? ", " : "") + "\"" + srnxr_counter_name(k) +
            "\": " + std::to_string(c[k]);
    }
    return js + "}";
  };

  CUDA_CHECK(cudaMemset(d_blob + hdr.counters_offset, 0, SRNXR_COUNTER_NUM * 8));
  const auto run_start = std::chrono::steady_clock::now();
  launch(n);
  CUDA_CHECK(cudaDeviceSynchronize());
  const double cold_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                      run_start).count();
  const std::string counters_json = read_counters();
  printf("full set: %d points, first launch %.3f ms, counters %s\n", n, cold_s * 1e3,
         counters_json.c_str());
  jkv("cold_launch_ms_full_set", std::to_string(cold_s * 1e3), false);
  jkv("counters_full_set", counters_json, false);
  std::vector<float> values(size_t(n) * output_dim);
  CUDA_CHECK(cudaMemcpy(values.data(), d_out, values.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  if (!batches.empty()) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::string bj = "[";
    for (size_t bi = 0; bi < batches.size(); bi++) {
      const int b = std::min(batches[bi], n);
      params.n = b;
      CUDA_CHECK(cudaMemcpy(d_params, &params, sizeof(params), cudaMemcpyHostToDevice));
      for (int r = 0; r < 3; r++) {
        OPTIX_CHECK(optixLaunch(pipeline, nullptr, CUdeviceptr(d_params), sizeof(params), &sbt,
                                unsigned(b), 1, 1));
      }
      std::vector<float> times(repeat);
      for (int r = 0; r < repeat; r++) {
        CUDA_CHECK(cudaEventRecord(e0));
        OPTIX_CHECK(optixLaunch(pipeline, nullptr, CUdeviceptr(d_params), sizeof(params), &sbt,
                                unsigned(b), 1, 1));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], e0, e1));
      }
      std::sort(times.begin(), times.end());
      const float median = times[times.size() / 2];
      printf("batch %7d: median %9.4f ms, best %9.4f ms, %10.3f us/query, %9.3f Mqueries/s\n",
             b, median, times.front(), median * 1e3 / b, b / (median * 1e3));
      char buf[256];
      snprintf(buf, sizeof(buf),
               "%s{\"batch\": %d, \"median_ms\": %.6f, \"best_ms\": %.6f, \"us_per_query\": %.6f, "
               "\"mqueries_per_s\": %.6f}",
               bi ? ", " : "", b, median, times.front(), median * 1e3 / b, b / (median * 1e3));
      bj += buf;
    }
    jkv("batches", bj + "]", false);
  }

  if (!out_path.empty()) {
    std::ofstream f(out_path, std::ios::binary);
    f.write((const char *)values.data(), values.size() * sizeof(float));
  }
  optixPipelineDestroy(pipeline);
  optixProgramGroupDestroy(groups[0]);
  optixProgramGroupDestroy(groups[1]);
  optixModuleDestroy(module);
  optixDeviceContextDestroy(ctx);
  return finish(0);
}
