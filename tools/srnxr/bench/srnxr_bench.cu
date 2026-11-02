/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Standalone correctness and performance executable of the SRNXR version 2 decoder.
 *
 * Builds the device blob with the Cycles loader code (src/scene/srnxr_model.cpp) and runs the
 * Cycles decoder (src/kernel/util/image_3d_srnxr_decode.h) on a fixed set of query points:
 *
 *   srnxr_bench model.srnxr points.f32 [options]
 *     --backend cuda_fused|cuda_tensor|cpu   decoder (cpu: host build of the CUDA_FUSED path)
 *     --sine accurate|fast                   sinf or __sinf
 *     --out values.f32                       decoded fields, float32 [N, output_dim]
 *     --batches 1,32,256,...                 timed launch sizes (prefixes of the point set)
 *     --repeat R                             timed launches per batch size
 *     --block B                              threads per block
 *     --no-counters                          do not update the device statistics
 *     --per-thread                           CUDA_FUSED: per-thread SIMT only (no warp path)
 *     --active-fraction F                    decode in two launches, each with a random fraction
 *                                            F (and 1 - F) of the threads active
 *     --json results.json                    machine readable results
 *
 * points.f32 holds float32 [N, 4] = (x, y, z, t). Without --batches the whole set is decoded
 * once (correctness run). */

#include "srnxr_compat.h"

#include "scene/srnxr_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
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

/* select: 0 every thread decodes; 1 or 2 only the threads in, or not in, a pseudo-random subset
 * of threshold / 1024 of the threads, the others return before the decoder, which leaves the
 * warps partially active as in the Cycles volume integrator. */
__global__ void decode_kernel(const uint8_t *blob,
                              const float4 *points,
                              const int n,
                              const int output_dim,
                              float *out,
                              const uint32_t select,
                              const uint32_t threshold)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (select) {
    const bool in_set = ((uint32_t(i) * 2654435761u) >> 22) < threshold;
    if (in_set != (select == 1)) {
      return;
    }
  }
  const bool active = i < n;
  float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (active) {
    const float4 q = points[i];
    p[0] = q.x;
    p[1] = q.y;
    p[2] = q.z;
    p[3] = q.w;
  }
  float v[SRNXR_V2_MAX_OUTPUTS];
  /* Every thread of the grid calls the decoder so warps stay converged for CUDA_TENSOR. */
  srnxr_v2_decode(blob, p, active, v);
  if (active) {
    for (int k = 0; k < output_dim; k++) {
      out[size_t(i) * output_dim + k] = v[k];
    }
  }
}

__global__ void empty_kernel() {}

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

static std::vector<int> parse_list(const std::string &s)
{
  std::vector<int> r;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    r.push_back(atoi(item.c_str()));
  }
  return r;
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

int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.srnxr points.f32 [options]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1], points_path = argv[2];
  std::string backend_name = "cuda_fused", sine_name = "accurate", out_path, json_path;
  std::vector<int> batches;
  int repeat = 20, block = 128;
  bool counters = true;
  /* CUDA_FUSED without the warp-cooperative copy: always the per-thread SIMT path. */
  bool per_thread = false;
  /* Partially active warps one query per thread. */
  bool no_partial = false;
  /* < 1: every launch runs as two launches over complementary random thread subsets */
  float active_fraction = 1.0f;
  for (int i = 3; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (a == "--backend") {
      backend_name = next();
    }
    else if (a == "--sine") {
      sine_name = next();
    }
    else if (a == "--out") {
      out_path = next();
    }
    else if (a == "--json") {
      json_path = next();
    }
    else if (a == "--batches") {
      batches = parse_list(next());
    }
    else if (a == "--repeat") {
      repeat = atoi(next().c_str());
    }
    else if (a == "--block") {
      block = atoi(next().c_str());
    }
    else if (a == "--no-counters") {
      counters = false;
    }
    else if (a == "--per-thread") {
      per_thread = true;
    }
    else if (a == "--no-partial") {
      no_partial = true;
    }
    else if (a == "--active-fraction") {
      active_fraction = float(atof(next().c_str()));
    }
    else {
      fprintf(stderr, "unknown option %s\n", a.c_str());
      return 1;
    }
  }

  const bool cpu = backend_name == "cpu";
  SRNXRBuildSettings settings;
  if (!cpu && !srnxr_backend_from_name(backend_name, settings.backend)) {
    fprintf(stderr, "unknown backend %s\n", backend_name.c_str());
    return 1;
  }
  if (settings.backend == SRNXR_BACKEND_COOPVEC) {
    fprintf(stderr, "COOPVEC runs in OptiX programs only, use srnxr_coopvec\n");
    return 1;
  }
  if (!srnxr_sine_mode_from_name(sine_name, settings.sine_mode)) {
    fprintf(stderr, "unknown sine mode %s\n", sine_name.c_str());
    return 1;
  }

  std::vector<char> file;
  if (!read_file(model_path, file)) {
    fprintf(stderr, "cannot read %s\n", model_path.c_str());
    return 1;
  }
  std::vector<char> pts_raw;
  if (!read_file(points_path, pts_raw) || pts_raw.size() % 16) {
    fprintf(stderr, "cannot read %s or its size is not a multiple of 16 bytes\n",
            points_path.c_str());
    return 1;
  }
  const int n = int(pts_raw.size() / 16);

  /* Time per timestep of the model is the point's t, the header time mapping is identity. */
  settings.has_time_frame = true;
  settings.time_frame = 0.0f;
  settings.time_shutter = 0.0f;

  std::vector<char> blob;
  SRNXRBuildReport report;
  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  if (!srnxr_build_device_blob_v2(file.data(), file.size(), settings, blob, report, error)) {
    fprintf(stderr, "SRNXR load failed: %s\n", error.c_str());
    return 1;
  }
  SRNXRDeviceHeaderV2 hdr;
  memcpy(&hdr, blob.data(), sizeof(hdr));
  if (!counters) {
    hdr.flags &= ~SRNXR_DEVICE_FLAG_COUNTERS;
  }
  if (per_thread) {
    hdr.flags &= ~SRNXR_DEVICE_FLAG_WARP_WEIGHTS;
  }
  if (no_partial) {
    hdr.flags &= ~SRNXR_DEVICE_FLAG_PARTIAL_WARPS;
  }
  /* The decode kernel passes the point time directly, not frame time + shutter offset. */
  memcpy(blob.data(), &hdr, sizeof(hdr));
  printf("SRNXR model %s\n%s", model_path.c_str(), report.summary().c_str());
  const int output_dim = int(hdr.output_dim);
  std::vector<float> values(size_t(n) * output_dim, 0.0f);

  std::string json = "{\n";
  auto jkv = [&](const char *key, const std::string &value, bool quote = true) {
    json += std::string("  \"") + key + "\": " + (quote ? "\"" + json_escape(value) + "\"" : value) +
            ",\n";
  };
  jkv("model", model_path);
  jkv("points", points_path);
  jkv("point_count", std::to_string(n), false);
  jkv("backend_requested", cpu ? "CPU_REFERENCE_SIMT_FP32" : srnxr_backend_name(settings.backend));
  jkv("sine_mode", srnxr_sine_mode_name(settings.sine_mode));
  jkv("per_thread_only", per_thread ? "true" : "false", false);
  jkv("active_fraction", std::to_string(active_fraction), false);
  jkv("synthetic_model", report.synthetic ? "true" : "false", false);
  jkv("load_report", report.summary());
  jkv("parse_seconds", std::to_string(report.parse_seconds), false);
  jkv("pack_seconds", std::to_string(report.pack_seconds), false);
  jkv("device_bytes", std::to_string(report.device_bytes), false);
  jkv("mma_weight_bytes", std::to_string(report.mma_weight_bytes), false);

  if (cpu) {
    const auto start = std::chrono::steady_clock::now();
    const int threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
      pool.emplace_back([&, t]() {
        for (int i = t; i < n; i += threads) {
          float p[4];
          memcpy(p, pts_raw.data() + size_t(i) * 16, 16);
          float v[SRNXR_V2_MAX_OUTPUTS];
          srnxr_v2_decode((const uint8_t *)blob.data(), p, true, v);
          memcpy(&values[size_t(i) * output_dim], v, sizeof(float) * output_dim);
        }
      });
    }
    for (std::thread &th : pool) {
      th.join();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("CPU decode of %d points with %d threads: %.3f s\n", n, threads, s);
    jkv("cpu_seconds", std::to_string(s), false);
    jkv("cpu_threads", std::to_string(threads), false);
  }
  else {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    int clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    printf("GPU %s, SM %d.%d, driver %d, runtime %d\n", prop.name, prop.major, prop.minor, driver,
           runtime);
    jkv("gpu", prop.name);
    jkv("sm", std::to_string(prop.major * 10 + prop.minor), false);
    jkv("cuda_driver_version", std::to_string(driver), false);
    jkv("cuda_runtime_version", std::to_string(runtime), false);
    jkv("gpu_clock_khz_max", std::to_string(clock_khz), false);
    jkv("mem_clock_khz_max", std::to_string(mem_clock_khz), false);
    if (settings.backend == SRNXR_BACKEND_CUDA_TENSOR && prop.major < 8) {
      fprintf(stderr, "CUDA_TENSOR needs SM 8.0 or newer (mma.m16n8k16), this GPU is SM %d.%d\n",
              prop.major, prop.minor);
      return 2;
    }

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, decode_kernel));
    printf("decode_kernel: %d registers, %zu bytes local, %zu bytes static shared, PTX %d, "
           "binary %d\n",
           attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, attr.ptxVersion,
           attr.binaryVersion);
    jkv("kernel_registers", std::to_string(attr.numRegs), false);
    jkv("kernel_local_bytes", std::to_string(attr.localSizeBytes), false);
    jkv("kernel_binary_version", std::to_string(attr.binaryVersion), false);
    jkv("block_size", std::to_string(block), false);

    const auto upload_start = std::chrono::steady_clock::now();
    uint8_t *d_blob = nullptr;
    float4 *d_points = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blob, blob.size()));
    CUDA_CHECK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    const double upload_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                          upload_start).count();
    jkv("upload_seconds", std::to_string(upload_s), false);
    jkv("load_seconds_total",
        std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start)
                           .count()),
        false);
    CUDA_CHECK(cudaMalloc(&d_points, pts_raw.size()));
    CUDA_CHECK(cudaMemcpy(d_points, pts_raw.data(), pts_raw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out, values.size() * sizeof(float)));

    const uint32_t threshold = uint32_t(active_fraction * 1024.0f + 0.5f);
    auto launch = [&](const int count) {
      const int grid = (count + block - 1) / block;
      if (active_fraction >= 1.0f) {
        decode_kernel<<<grid, block>>>(d_blob, d_points, count, output_dim, d_out, 0, 0);
      }
      else {
        decode_kernel<<<grid, block>>>(d_blob, d_points, count, output_dim, d_out, 1, threshold);
        decode_kernel<<<grid, block>>>(d_blob, d_points, count, output_dim, d_out, 2, threshold);
      }
    };
    auto reset_counters = [&]() {
      CUDA_CHECK(cudaMemset(d_blob + hdr.counters_offset, 0, SRNXR_COUNTER_NUM * sizeof(uint64_t)));
    };
    auto read_counters = [&](std::string &js) {
      uint64_t c[SRNXR_COUNTER_NUM];
      CUDA_CHECK(cudaMemcpy(c, d_blob + hdr.counters_offset, sizeof(c), cudaMemcpyDeviceToHost));
      js = "{";
      for (int k = 0; k < SRNXR_COUNTER_WARP_ACTIVE_LANES + 1; k++) {
        js += std::string(k ? ", " : "") + "\"" + srnxr_counter_name(SRNXRCounter(k)) +
              "\": " + std::to_string(c[k]);
      }
      js += "}";
      return c[SRNXR_COUNTER_FALLBACK_SIMT];
    };

    /* Correctness run over all points, with counters. */
    reset_counters();
    const auto run_start = std::chrono::steady_clock::now();
    launch(n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double cold_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                        run_start).count();
    CUDA_CHECK(cudaMemcpy(values.data(), d_out, values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::string counters_json;
    const uint64_t fallback = read_counters(counters_json);
    printf("full set: %d points, first (cold) launch %.3f ms, counters %s\n", n, cold_s * 1e3,
           counters_json.c_str());
    jkv("cold_launch_ms_full_set", std::to_string(cold_s * 1e3), false);
    jkv("counters_full_set", counters_json, false);
    jkv("backend_actual",
        (settings.backend == SRNXR_BACKEND_CUDA_TENSOR && fallback) ?
            "CUDA_TENSOR with SIMT fallback evaluations (see counters)" :
            srnxr_backend_name(settings.backend));

    /* Empty launch cost */
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    {
      for (int r = 0; r < 10; r++) {
        empty_kernel<<<1, 32>>>();
      }
      CUDA_CHECK(cudaEventRecord(e0));
      for (int r = 0; r < 1000; r++) {
        empty_kernel<<<1, 32>>>();
      }
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      printf("empty kernel launch: %.2f us\n", ms);
      jkv("empty_launch_us", std::to_string(ms), false);
    }

    /* Timed batches, warm (after the correctness run). */
    if (!batches.empty()) {
      std::string bj = "[";
      for (size_t bi = 0; bi < batches.size(); bi++) {
        const int b = std::min(batches[bi], n);
        if (b <= 0) {
          continue;
        }
        for (int r = 0; r < 3; r++) {
          launch(b);
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> times(repeat);
        for (int r = 0; r < repeat; r++) {
          CUDA_CHECK(cudaEventRecord(e0));
          launch(b);
          CUDA_CHECK(cudaEventRecord(e1));
          CUDA_CHECK(cudaEventSynchronize(e1));
          CUDA_CHECK(cudaEventElapsedTime(&times[r], e0, e1));
        }
        std::sort(times.begin(), times.end());
        const float median = times[times.size() / 2];
        const float best = times.front();
        reset_counters();
        launch(b);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::string cj;
        read_counters(cj);
        printf("batch %7d: median %9.4f ms, best %9.4f ms, %10.3f us/query, %9.3f Mqueries/s\n",
               b, median, best, median * 1e3 / b, b / (median * 1e3));
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "%s{\"batch\": %d, \"median_ms\": %.6f, \"best_ms\": %.6f, \"us_per_query\": %.6f, "
                 "\"mqueries_per_s\": %.6f, \"counters\": ",
                 bi ? ", " : "", b, median, best, median * 1e3 / b, b / (median * 1e3));
        bj += buf + cj + "}";
      }
      bj += "]";
      jkv("batches", bj, false);
    }
    cudaFree(d_blob);
    cudaFree(d_points);
    cudaFree(d_out);
  }

  if (!out_path.empty()) {
    std::ofstream f(out_path, std::ios::binary);
    f.write((const char *)values.data(), values.size() * sizeof(float));
  }
  json += "  \"output_dim\": " + std::to_string(output_dim) + "\n}\n";
  if (!json_path.empty()) {
    std::ofstream f(json_path);
    f << json;
  }
  return 0;
}
