# Neural Volume Decoding in Blender Cycles — Implementation Specification

## Role and objective

Implement a production-oriented, testable extension to **Blender Cycles** for on-demand reconstruction of compressed scientific 4D CFD fields using local neural networks. Work directly in the existing repository; inspect its current source tree and build system before editing. The target is NVIDIA RTX Ada (RTX 4090) and RTX Blackwell, with Blackwell as the primary optimization target.

**Implement and compare exactly three selectable inference backends:**

1. **OptiX Cooperative Vectors (COOPVEC)**: FP16 matrix weights, cooperative-vector matrix–vector operations, custom sine activation for SIREN. Explicitly measure the effect of intermediate FP16 rounding.
2. **Fused CUDA scalar/SIMT MLP (CUDA_FUSED)**: FP16 stored weights, FP32 arithmetic and FP32 sine activation; no Tensor Core requirement. This is the portable and numerically stronger baseline.
3. **Fused CUDA Tensor Core MLP (CUDA_TENSOR)**: FP16 weights and Tensor Core matrix operations (WMMA, MMA, CUTLASS, or another justified mechanism), FP32 accumulation where supported and FP32 sine evaluation where achievable; keep layers fused as far as practical.

The names above are **logical configuration values**, not a claim that the same device function can be compiled into both CUDA and OptiX programs. No silent substitution of a different implementation is permitted: log the selected and actual backend.

## Source material and boundaries

The scientific method is based on the supplied one-page HPCSE 2026 poster, *Adaptive Domain Decomposition for Neural Surrogate Model Compression*, by Marcel Smolík and Tomáš Brzobohatý:

`25_HPCSE26_Marcel-Smolík_Tomas_Brzobohaty.pdf`

Public reference: https://hpcse.it4i.cz/HPCSE26/files/posters/25_HPCSE26_Marcel-Smol%C3%ADk_Tomas_Brzobohaty.pdf

Follow the **actual trained model weights and data-export schema** as ground truth when available. Do not invent missing architectural or normalization details. The poster includes inconsistencies (e.g., SIREN output shown as 1 in one layer diagram but also described as a five-component field). Identify these and implement validation checks instead of silently choosing one interpretation.

### Method described in the poster

- 4D query coordinate `(x,y,z,t)`.
- Adaptive octree partitioning, independent local models per leaf/subdomain.
- Spatial overlap: each strict domain extends by 15% in the indicated directions; verify the exact convention and time-axis treatment in the dataset/exporter.
- Normalize coordinates to the local block; predict with each relevant network; denormalize predictions; combine with Gaussian distance weights.
- Weighted reconstruction: `u(x,t) = sum_i(w_i(x) * u_i(x,t)) / sum_i(w_i(x))` with `w_i(r) = exp(-r*r/(2*sigma_i*sigma_i))` (confirm distance normalization and sigma from original implementation).
- Fields include velocity, pressure and density; five outputs are consistent with `(ux,uy,uz,p,rho)` but must be confirmed against actual checkpoints.
- SIREN: 4-coordinate input, hidden width 256, three/four hidden layers in reported variants, sine activation and `w0=30`; diagrams differ on output width.
- Fourier MLP: 4-coordinate input, 32 frequencies per coordinate, 260 encoded inputs, hidden width 32, GELU activation, five outputs. Frequency definition shown as `2^k*pi`, `k=0..31`.
- Poster reference throughput (not a guaranteed performance target): SIREN-3 4.77 Mpoints/s; SIREN-4 4.04 Mpoints/s; Fourier-3 4.59 Mpoints/s; Fourier-4 3.58 Mpoints/s. Hardware, batching and measurement scope are not documented sufficiently for direct comparison.
- Compression ratios reported are approximately 89.13x, 64.41x, 48.51x and 42.30x for these four configurations.

**IMPORTANT:** Implement SIREN first across all three backends. Preserve a decoder interface capable of handling Fourier models later, but do not claim the Fourier backend is supported before it is actually implemented and tested.

## Architectural constraints and research gate

1. Inspect the checked-out Blender/Cycles version, actual GPU device dispatch, shader/volume data paths, OptiX program types and applicable compiler flags. Record exact verified source filenames and line/function locations in the implementation report. Do not assume filenames from an older Blender release.
2. Cycles uses both CUDA kernels and OptiX components. **OptiX `optixCoopVecMatMul` is not automatically callable from arbitrary CUDA kernels.** Verify whether the requested volume sample execution can legally and efficiently occur in an OptiX program. The OptiX mode may require a separate OptiX execution path and integration changes in the wavefront scheduler.
3. Build a small standalone OptiX cooperative-vector capability/compilation probe before changing Cycles scheduling. Check `OPTIX_DEVICE_PROPERTY_COOP_VEC` and confirm supported matrix shapes, layouts, data types and execution context with the installed SDK.
4. If truly inline cooperative-vector sampling is incompatible with the current Cycles volume sampling path, **do not disguise CUDA inference as COOPVEC**. Implement a documented OptiX query-batch stage only if correct integration with Cycles state/queues can be established; otherwise mark the backend as experimental/unavailable in the Cycles integration while retaining the standalone executable and benchmarks. Continue implementing the other two backends.
5. Do not introduce a second ray tracer or replace the path integrator wholesale. Reuse Cycles' volume integration and shading semantics.
6. The source poster does not define optical transfer functions or a guaranteed extinction majorant. Treat them as independent integration requirements, not properties of the trained decoder.

## Backend selection: compile-time + runtime

Implement **both** build-time feature gates and a runtime backend selector. One macro alone is not sufficient because support depends on build configuration, GPU capability and OptiX program availability.

Suggested compile flags (adapt names to repository conventions):

```cmake
WITH_CYCLES_NEURAL_VOLUME=ON
WITH_CYCLES_NEURAL_COOPVEC=ON
WITH_CYCLES_NEURAL_CUDA_FUSED=ON
WITH_CYCLES_NEURAL_CUDA_TENSOR=ON
```

Suggested compile definitions / conditional compilation:

```cpp
WITH_CYCLES_NEURAL_VOLUME
WITH_CYCLES_NEURAL_COOPVEC
WITH_CYCLES_NEURAL_CUDA_FUSED
WITH_CYCLES_NEURAL_CUDA_TENSOR
```

Runtime enum (illustrative, not existing Cycles code):

```cpp
enum class NeuralInferenceBackend : uint8_t {
  CUDA_FUSED  = 0,
  CUDA_TENSOR = 1,
  COOPVEC     = 2,
};
```

Use a session/device-level runtime option (CLI/config first; Blender UI optional after correctness). Make sure a backend change is applied at a safe synchronization point and causes any required GPU pipeline or kernel reconfiguration. The selected mode must be encoded in logs, benchmark files and reproducibility metadata. An `AUTO` policy can be added later but **must not obscure measured results**.

On unsupported hardware, unsupported OptiX contexts, absent model formats or disabled build flags: report a clear error, or use an explicitly configured fallback with a warning. No implicit fallback in benchmark mode.

## GPU data format

Design versioned, validated binary serialization/export from reference training software:

- Header: magic, schema version, byte order, model architecture, field names/order, layer dimensions, activation, `w0`, numerical format, normalization metadata, scale/offset and checksum.
- Octree: compact node table, child indices, parent/leaf flags, bounds; separate overlap-aware leaf association or neighborhood lookup.
- Leaf metadata: strict bounds, expanded bounds, center, coordinate normalization, Gaussian parameters, network ID, matrix/bias offsets, output normalization.
- Weights/biases: contiguous, aligned GPU memory with explicit per-layer descriptors; separate backend-specific packing where required by OptiX or MMA.
- All offset arithmetic checked for overflow and bounds; no raw untrusted pointer/offset use.
- Validate no missing network references, malformed node topology, zero extents, invalid sigmas or incompatible layer widths.
- Cache or preload weights to avoid allocations or transfers during individual sample evaluation; account for model residency and GPU memory budgets.
- Shared reference representation: original FP32 master weights + deterministic FP16 conversion and backend packing; comparisons use the same quantized FP16 values unless intentionally testing quantization error.

Separate **decoder model format** from **backend-specific packed weight layouts**. Build an offline packing utility and on-device buffer upload path. Record packing time and memory overhead independently of inference time.

## Common decoding and sampling behavior

Expose a logical interface such as:

```cpp
struct NeuralFieldSample {
  float velocity[3];
  float pressure;
  float density;
};

// Illustrative host/device contract, not a promise of identical implementation
// or an ABI shared verbatim between OptiX and CUDA.
NeuralFieldSample sample_neural_volume(
    const NeuralVolumeView& volume,
    float3 object_space_position,
    float simulation_time,
    NeuralInferenceBackend backend);
```

- Use a **single mathematical specification** for local coordinate transforms, overlap detection, normalization, denormalization, Gaussian blending and output field mapping.
- Implement a CPU/PyTorch FP32 reference decoder to serve as the numerical oracle. The three GPU implementations must match the same mathematical semantics.
- Preserve time mapping between Blender frame/shutter time and simulation time as an explicit user-defined mapping.
- Support samples falling into multiple overlapping leaf models; normalize the sum of weights stably. Define exact behavior for outside-volume samples and unexpectedly zero weight sums.
- Avoid evaluating adjacent networks when their influence is provably zero; do not truncate Gaussian tails without specifying and validating the approximation.
- Use FP32 for positions, blending, Gaussian weights, output denormalization, and final material/volume properties; evaluate alternative precision only as an experiment.
- SIREN forward pass: `h_0 = x_local`, `h_{l+1} = sin(w0_l * (W_l h_l + b_l))` with **actual checkpoint frequency convention**; do not assume `w0=30` applies to every layer unless confirmed. Final layer typically has no sine activation, but verify from checkpoint/source.
- Keep consistent order of operations where reproducible results are required; document unavoidable differences from FMA, math library approximations or reassociation.

## Backend A — OptiX Cooperative Vectors (COOPVEC)

- Verify installed OptiX SDK supports `OptixCoopVec`, `optixCoopVecMatMul`, matrix layout conversion and the requested execution program. Use NVIDIA SDK examples as reference.
- Target FP16 matrix weights and supported FP16 input/output forms. **Do not assume native FP32 results for FP16 cooperative-vector matmul.** Document each conversion and intermediate rounding.
- Implement the SIREN sine activation with the supported vector operations and/or per-component `sinf` conversion path as allowed by the API. There is no need to approximate sine if a legal and sufficiently efficient full-precision conversion is available; benchmark both numeric accuracy and cost.
- No claim that a native `optixCoopVecSin` exists. Confirm that reading/writing vector elements and conversion to FP32 is legal in the chosen program type.
- Pack weight matrices in the inference-optimal OptiX layout using the installed SDK's supported conversion mechanisms. Validate alignment, lifetimes and layout-specific byte sizes.
- The key question is performance when nearby ray samples address **different leaf models/weight matrices**. Measure warp coherence and divergence.
- Prove that integration runs real cooperative-vector instructions, not a hidden SIMT fallback. Unsupported configurations must fail explicitly.
- Isolate OptiX pipeline-specific code behind an adapter and provide an independent validation executable.

## Backend B — fused CUDA SIMT MLP (CUDA_FUSED)

- Store weights in FP16, convert to FP32 for multiply/accumulate and FP32 sine.
- Implement minimal launches and keep one local network's hidden activations within thread/warp registers and shared memory where practical.
- Choose one sample per thread or warp according to measured instruction count, register pressure, occupancy and model width; document the actual execution mapping.
- Specialize layer widths/configurations (`4 -> 256 -> ... -> out` initially). Avoid generic dynamic loops in the inner kernel if specialization is faster.
- Control register spilling. Benchmark `sinf` and `__sinf` only if the latter's error is acceptable for the physics dataset; do not silently use fast-math.
- This backend is the reference for separating storage-quantization error from cooperative-vector intermediate-rounding error.

## Backend C — fused CUDA Tensor Core MLP (CUDA_TENSOR)

- Use a genuine Tensor Core path such as CUDA MMA/WMMA or a relevant CUTLASS/CuTe implementation; validate actual generated instructions/profiler counters.
- FP16 weight storage; FP32 accumulation for matrix multiplications where the instruction supports it; explicitly specify input activation type, casts and any FP16 rounding between layers.
- Group samples belonging to the same network for effective GEMM tiles if necessary. Such grouping may require queues/reordering; account for sorting, dispatch and scatter costs in end-to-end benchmarks.
- Favor fused layers only where resource limits permit. If a fully fused implementation underperforms or spills significantly, use a justified hybrid and report the extra launches.
- RTX 4090 (SM 8.9) and RTX Blackwell (target SM confirmed at configure time) require different kernel specializations. Never compile Blackwell-only PTX for RTX 4090.
- Implement and benchmark explicit handling of odd output widths, layer padding and small batch occupancy.

## Scientific correctness and precision tests

Create automated comparison tests using representative exported model weights and overlapping octree leaves. If no real checkpoints are present, create a deterministic synthetic surrogate **and mark such tests as synthetic**; do not claim dataset fidelity.

Compare at least:

1. Full-FP32 reference model against FP16-quantized FP32 CUDA evaluation (storage error).
2. FP16-quantized FP32 CUDA versus CUDA Tensor Core (arithmetic error).
3. FP16-quantized FP32 CUDA versus Cooperative Vectors (arithmetic and intermediate-rounding error).
4. Whole-field reconstruction including octree traversal, overlapping leaves, Gaussian blending and denormalization.

Required test locations: block centers, strict boundaries, 15% overlap zones, octree T-junctions, high-gradient flow regions, time-boundary samples and points just outside volume bounds. Use deterministic seeds and fixed coordinate sets.

Required metrics: per-field MAE, RMSE/NRMSE, P99 and max absolute error, relative error with a sensible near-zero policy, PSNR, gradient error, isosurface deviation, boundary continuity, temporal consistency, and errors in any derived quantity used for rendering. Establish precision acceptance limits from **real dataset requirements** rather than inventing one percentage. Save error maps for representative slices.

Special attention:

- SIREN `sin(w0*x)` can amplify phase errors; analyze errors layer by layer.
- Test both `sinf` and fast approximate sine as separate configurations where available.
- Fourier variant uses frequencies up to `2^31*pi`; coordinate rounding and phase reduction may be critical. Keep Fourier encoding in FP32 at minimum and evaluate double-precision phase construction/reference where warranted.
- Do not conflate neural approximation error with *additional* quantization/backend error.

## Cycles integration

Plan the Cycles modifications in two stages.

### Stage 1: functional CUDA integration

- Add a neural-volume resource loader and scene/device representation to the existing Cycles architecture.
- Insert a decoder-backed volume attribute sampling path at the **verified current code location** that normally fetches volume scalar/vector attributes. Keep existing volume material and path integration behavior.
- The field-to-optics transfer function (density/absorption/scattering/emission/color) belongs in the material layer, not in the neural network weights. Expose at least density and one additional scalar property for validation.
- Verify object/world coordinate transforms, interpolation/time, volume bounds, volume ray steps, camera motion and scene updates.
- Handle volume majorants conservatively: if the neural decoder cannot provide certified extinction bounds, precompute conservative bound metadata with verification, use a conservative global bound, or disable optimizations that depend on a bound until correctness is established. Never derive an unsafe majorant merely by sampling a few points.
- Avoid the trap of reconstructing the entire field into a grid on every frame. The intended path is **on-demand samples**.

### Stage 2: real OptiX COOPVEC integration

- Document the host/device boundary of the current volume sampling path and the exact conditions under which an OptiX program can evaluate cooperative vectors.
- Implement an OptiX sampling path only when the required Cycles ray state and material data can be made available correctly. Preserve RNG, path state and volume integration semantics.
- If a separate OptiX neural-query pass is chosen, explicitly schedule query generation, GPU buffer exchange, execution, completion and result consumption; avoid illegal concurrent writes or implicit synchronizations.
- If genuine OptiX integration would require excessive scope or architectural changes, leave it isolated behind the feature gate and report the blocker, while delivering the working CUDA backends and standalone COOPVEC benchmark.
- Maintain regression compatibility for existing non-neural volumes and all other supported Cycles GPU paths. Unsupported render devices should fail gracefully or disable the feature.

## Benchmark methodology

Measure separately and end to end on RTX 4090 and RTX Blackwell if both are available. Otherwise record only hardware that was actually tested. Include GPU model, driver, CUDA and OptiX SDK versions, SM, clocks/power settings (if available), compiler flags and Cycles commit.

Test batch/sample counts: `1, 32, 256, 1024, 4096, 16384, 65536` where meaningful. Measure microseconds/query at small batch, Mqueries/s at large batch, kernel launch cost, model packing/load time, residency, memory bandwidth, registers, occupancy, Tensor Core utilization, SINE time and octree lookup time.

Create workloads:

- All queries share one leaf model (best case).
- Queries belong to a few neighboring models (spatial coherence).
- Queries access random models (worst-case divergence/cache).
- Queries lie in overlapping regions and invoke multiple networks.
- Realistic per-ray volume sampling with varying step counts, scene geometry and mixed shading paths.

Benchmark both SIREN 3- and 4-hidden-layer configurations, **after confirming exact layer definitions in checkpoints**. Include whole-frame time, rays/s, samples/s, memory footprint, resulting image differences and volume-only breakdown. Report both cold and warm cache behavior. Do not compare raw Tensor Core peak FLOPS with end-to-end decoder throughput as if they were interchangeable.

Generate machine-readable CSV/JSON and a concise Markdown report including accuracy-versus-speed Pareto plots. Explicitly state which backends were genuinely executed. Never fabricate benchmarks or fill unknown measurements with estimates.

## Deliverables

1. Implemented source modifications integrated into the checked-out Cycles tree.
2. Configurable backend selection (compile gates + runtime enum) with accurate diagnostics.
3. GPU model loader/packer and stable versioned model serialization (or a documented minimal import contract compatible with real model exports).
4. Standalone correctness and performance executables for all feasible backends.
5. Functional CUDA neural-volume rendering path in Cycles.
6. OptiX Cooperative Vectors integration if confirmed feasible; otherwise a working standalone implementation plus an explicit technical integration plan/blocker.
7. Unit/integration tests, deterministic test fixtures, precision comparison scripts and benchmark scripts.
8. `docs/neural_volume.md`: architecture, exact modified files, build flags, runtime selection, mathematical conventions, memory layout, limitations and reproduction steps.
9. `docs/neural_volume_results.md`: verified hardware/software, benchmark tables, errors, model variants and open issues. Mark untested combinations as **NOT TESTED**.

## Execution order and acceptance gates

**Gate 0 — Repository reconnaissance**: inspect existing Cycles, CUDA and OptiX source paths; build toolchains; locate volume sampling code. Document findings before edits.

**Gate 1 — Reference decoder**: real or synthetic model import, correct octree overlap, normalization, SIREN evaluation and Gaussian blending against CPU/PyTorch reference.

**Gate 2 — CUDA_FUSED**: working GPU decoder integrated into Cycles and matched to FP32-reference outputs within justified limits.

**Gate 3 — CUDA_TENSOR**: verified Tensor Core instructions, performance comparison, accuracy comparison, with an honest accounting of batching/reordering overhead.

**Gate 4 — COOPVEC**: standalone compiled/running implementation, real device capability verification, per-layer sine tests and numerical comparisons.

**Gate 5 — Cycles COOPVEC path**: integration only if the actual execution context supports it; otherwise produce a precise blocker and do not claim it works.

**Gate 6 — End-to-end validation**: real render scene, majorant correctness, coverage of overlaps, all relevant backend modes, regression checks, CSV metrics and documentation.

Proceed incrementally. After each gate, build and run the available tests, report completed work and current limitations, then continue. Do not rewrite unrelated Cycles subsystems. Do not pretend a backend is optimized or available unless compilation, runtime execution and profiler evidence demonstrate it. No fabricated measurements, API functions, or source locations.

## Key technical references

- Source poster: https://hpcse.it4i.cz/HPCSE26/files/posters/25_HPCSE26_Marcel-Smol%C3%ADk_Tomas_Brzobohaty.pdf
- Blender Cycles source layout: https://developer.blender.org/docs/features/cycles/source_layout/
- Blender repository: https://projects.blender.org/blender/blender
- NVIDIA OptiX Cooperative Vectors: https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/
- OptiX cooperative-vector API: https://raytracing-docs.nvidia.com/optix9/api/group__optix__device__api__coop__vec.html
- NVIDIA tiny-cuda-nn (implementation reference, not a mandatory dependency): https://github.com/NVlabs/tiny-cuda-nn

