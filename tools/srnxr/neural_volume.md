# Neural volumes in Cycles (SRNXR format 2)

On-demand decoding of compressed 4D fields stored as an adaptive octree of small local SIREN
networks, inside the Cycles volume integrator. This document describes the format, the
mathematics, the three inference backends, the Cycles integration, the build and run options and
the limitations. Measurements are in [neural_volume_results.md](neural_volume_results.md).

The method follows the HPCSE 2026 poster *Adaptive Domain Decomposition for Neural Surrogate
Model Compression* (Smolík, Brzobohatý). No checkpoints of that work were available: every model
used here is the **synthetic** surrogate of `tools/srnxr/srnxr_make_synthetic.py`.

## Overview

| part | files |
|---|---|
| file format, writer and reader (Python) | `tools/srnxr/srnxr_format.py` |
| float64 reference decoder (numpy oracle) | `tools/srnxr/srnxr_reference.py` |
| synthetic surrogate generator (torch training) | `tools/srnxr/srnxr_make_synthetic.py` |
| test points, comparison, field analysis, bound check, report | `tools/srnxr/srnxr_points.py`, `srnxr_compare.py`, `srnxr_analysis.py`, `srnxr_bounds_check.py`, `srnxr_report.py` |
| test scene generator | `tools/srnxr/srnxr_make_scene.py` |
| host parser, validator and device packer | `src/scene/srnxr_model.{h,cpp}` |
| device structures | `src/kernel/util/image_3d_srnxr.h` |
| device decoder (CPU, CUDA, OptiX) | `src/kernel/util/image_3d_srnxr_decode.h` |
| Cycles image loader | `SRNXRImageLoader` in `src/scene/image_vdb.{h,cpp}` |
| Cycles kernel entry | `kernel_image_interp_srnxr_v2()` in `src/kernel/util/image_3d.h` |
| OptiX volume programs (COOPVEC) | `src/kernel/device/optix/kernel_neural_volume.cu` |
| standalone benchmark and validation | `tools/srnxr/bench/srnxr_bench.cu` (CUDA), `srnxr_coopvec.cu` (OptiX) |

Format 1 files (3D, one output, uniform grid; the previous `.srnxr`) are still read by the same
loader and decoded by the unchanged format 1 kernel path.

## Differences to the previous implementation

The format 1 loader and kernel did not match the method of the poster:

| topic | format 1 | format 2 |
|---|---|---|
| query | (x, y, z) | (x, y, z, t), t from the frame time and the ray time |
| outputs | 1 (density) | up to 8 named fields (`ux uy uz p rho`) |
| leaf lookup | uniform CSR grid over a modulo-shifted domain | adaptive octree, per-leaf neighbor lists computed by the loader |
| support | `max abs(xi) <= overlap` in block units | explicit strict and expanded (15 %) boxes per leaf |
| blending | global sigma, block-normalized distance | per-leaf center, axis scales and sigma; max-shifted normalization |
| w0 | one value | one per sine layer |
| width | <= 64 (kernel scratch) | <= 256, specialized 64 / 128 / 256 |
| weights on device | FP32 converted from FP16 | FP16 plus backend packings (mma fragments, transposed, OptiX layout) |
| denormalization | per block mu, sigma and a global affine, then clamp to [0, 1] | per leaf and field scale and offset, no clamp (the transfer function is the material) |
| empty weight sum | density 0 below an epsilon | cannot happen inside the domain (the strict leaf always contributes) |
| sine | `__sinf` always | `sinf` or `__sinf`, selectable, accurate even with `--use_fast_math` |
| backends | one SIMT path | CUDA_FUSED, CUDA_TENSOR, COOPVEC with compile and run time selection |
| majorants | Cycles samples 16 points per voxel of the volume octree | certified bounds from the weights |
| validation | layout and CSR | + topology, coverage, field names, w0, sigma, finite values, offsets |

Inconsistencies of the poster that the format leaves to the data instead of choosing one:
SIREN output width (1 in a diagram, five fields in the text): `output_dim` and `FIELD_NAMES`;
w0 = 30 for all layers or only the first: `LAYER_W0`; three or four hidden layers:
`hidden_layer_count`; the overlap convention and the treatment of time: explicit strict and
expanded boxes (the loader reports the measured expansion); the Gaussian distance normalization
and sigma: explicit per-leaf center, axis scales and sigma; a sine on the output layer: a header
flag (only the linear head is supported, the loader rejects the other). The Fourier MLP variant has
an architecture code but is rejected by the loader (not implemented).

## File format 2

Little-endian container shared with format 1: fixed header, section table (64 bytes per section:
name[32], dtype, element count, byte offset, byte length, CRC-32), 64-byte aligned sections,
header CRC and whole-file CRC (with its own field zeroed). Magic `SRNXR\0\1\0`, `format_version`
2, header 124 bytes padded to 128.

Header: `architecture` (1 SIREN, 2 Fourier MLP reserved), `input_dim` (3 or 4), `output_dim`,
`hidden_layer_count`, `hidden_width`, `network_count`, `leaf_count`, `node_count`,
`weights_dtype` (FP16 or FP32), `flags` (bit 0 linear head, bit 1 FP32 master weights present),
`domain_min[4]`, `domain_max[4]`.

| section | type, shape | content |
|---|---|---|
| `FIELD_NAMES` | u8 | `output_dim` NUL terminated unique names |
| `LAYER_W0` | f32 [L] | w0 of every sine layer |
| `NODES` | u32 [nodes, 4] | first child (8 consecutive children, or invalid), leaf index, parent, depth |
| `NODE_BOUNDS` | f32 [nodes, 6] | min xyz, max xyz |
| `LEAF_NODE` | u32 [leaves] | octree node of every leaf |
| `LEAF_STRICT_BOUNDS` | f32 [leaves, 8] | min xyzt, max xyzt |
| `LEAF_EXPANDED_BOUNDS` | f32 [leaves, 8] | support of the leaf network |
| `LEAF_INPUT_TRANSFORM` | f32 [leaves, 8] | offset xyzt, scale xyzt: `x = (p - offset) * scale` |
| `LEAF_BLEND` | f32 [leaves, 9] | center xyzt, axis scale xyzt, sigma |
| `LEAF_NETWORK` | u32 [leaves] | network of the leaf (networks may be shared) |
| `LEAF_OUTPUT_TRANSFORM` | f32 [leaves, 2 out] | scale, offset per field: `u = y * scale + offset` |
| `NETWORK_LAYOUT` | u32 [networks, 4] | width, weight offset, bias offset (elements), 0 |
| `SIREN_WEIGHTS` | f16 or f32 | per network: [W][in], (L - 1) x [W][W], [out][W], row-major |
| `SIREN_BIASES` | f32 | per network: L x W, out |
| `SIREN_WEIGHTS_FP32` | f32, optional | unquantized master weights |
| `MODEL_METADATA_JSON` | json, optional | provenance; `"synthetic": true` is reported by the loader |

The loader validates the checksums, sizes and offsets of every section with 64-bit arithmetic,
the octree (every node reached once from the root, 8 children that are the octants of the parent,
leaves owning exactly one leaf record), leaf boxes (strict = node box, expanded contains strict,
time-partitioned leaves are rejected), non-zero input scales, positive sigmas, existing networks,
finite FP16 weights and biases. FP32 weights are rounded to FP16 (round to nearest even) by the
loader and reported.

## Decoding contract

`srnxr_v2_decode()` (`image_3d_srnxr_decode.h`), independently implemented by the oracle
`srnxr_reference.decode()`:

1. `p = (P, t)`, P in object space, `t = time_frame + time_shutter * (ray_time - 0.5)` (the
   simulation time of the frame and the shutter length are image settings).
2. Outside the closed domain box every field is `outside_value` (default 0).
3. Strict leaf L: octree descent; a coordinate equal to a node center goes to the upper child.
4. For every leaf j of `neighbors(L)` (leaves whose expanded box intersects the strict box of L,
   ascending index, computed by the loader) with p in the closed expanded box of j:
   `x = (p - offset_j) * scale_j`; `h = sin(w0_l (W_l h + b_l))` for the L sine layers;
   `u_j = (W_head h + b_head) * out_scale_j + out_offset_j`;
   `e_j = -|(p - c_j) * s_j|^2 / (2 sigma_j^2)`.
5. `u = sum exp(e_j - m) u_j / sum exp(e_j - m)` with the running maximum m of the exponents,
   so the weight sum is >= 1 (the strict leaf contributes) and never underflows.

The support truncation of the Gaussian weights at the expanded box is part of the method; its
discontinuity is measured (`srnxr_analysis.py`, "support jump"). No further truncation is done.
Blending, positions, denormalization and the decoded fields are FP32 in every backend.

## Device blob

`srnxr_build_device_blob_v2()` packs: header (`SRNXRDeviceHeaderV2`: dimensions, flags, backend,
sine mode, channel mapping, domain, time mapping, majorant bake margin, outside value, `w0` per
layer, offsets), octree nodes, leaves (input / blend / output transforms, neighbor range,
certified bounds of the mapped channels), neighbor lists, networks, FP16 weights (row-major,
every network 16-byte aligned), FP32 biases, and depending on the backend: the
`mma.m16n8k16` B fragment copy (CUDA_TENSOR), the transposed copy of the hidden layers
(CUDA_FUSED and CUDA_TENSOR on GPUs), the cooperative vector region with its row-major staging
copy and matrix table (COOPVEC), and 16 statistics counters. Parse and pack times are reported
separately from inference.

An image returns `(u[f0], u[f1], u[f2], 1)` for three selected fields or `(u, u, u, 1)` for one.
Every SRNXR attribute is its own image and blob.

## Backends

All three evaluate the same networks from the same FP16 weights; the first layer (4 inputs) is
FP32 everywhere.

**CUDA_FUSED** (also the CPU path). FP16 weights, FP32 multiply-accumulate in index order, FP32
sine. Three execution mappings with identical sums:
- one query per thread (`srnxr_v2_eval_network_simt`): CPU, and GPU warps with fewer than 8
  active lanes;
- warp-cooperative (`srnxr_v2_eval_network_warp`, fully active warps): lane l owns neurons
  `64 j + 2 l + {0, 1}`, inputs broadcast by shuffle, 4 queries per pass, transposed FP16
  weights read coalesced; the output layer is a warp reduction;
- partial warps (`srnxr_v2_eval_network_partial`, 8 - 31 active lanes): activations of 2 queries
  in a per-warp shared memory slice (25 KB per 384-thread block), contiguous neuron pairs per
  active lane, 16-byte weight loads.

**CUDA_TENSOR**. Queries of a fully active warp that need the same network form 16-row tiles of
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (SM 8.0+). Activations FP16 in A fragment
layout in registers (the FP32 accumulators of two adjacent output tiles are exactly the A
fragment of the next layer), FP32 accumulation, FP32 bias and sine, FP16 rounding once per
layer. Weights pre-packed in B fragment order, all fragments of an output tile pair are loaded
before the first mma. Partially active warps use the partial-warp SIMT path and are counted as
`fallback_simt`.

**COOPVEC**. `optixCoopVecMatMul` with FP16 matrices in `INFERENCING_OPTIMAL` layout (converted
on the device by `optixCoopVecMatrixConvert` from the row-major staging copy) and FP16 vectors,
one query per thread inside OptiX programs. OptiX 9.1 rejects FP32 output vectors for FP16
inputs, so the layer outputs and biases are FP16; there is no cooperative vector sine, the sine
is applied per element in FP32. Only in OptiX programs: in Cycles the volume shading kernels
then run as ray generation programs (`kernel_neural_volume.cu`) and the network evaluation is
compiled once, as three direct callables (widths 64, 128, 256) the programs call by SBT index;
inlined into every call site of the volume shader evaluation the OptiX module did not finish
compiling in 15 minutes (60 GB), as callables the first compile takes about 3 minutes and is
cached by the driver.

No backend is replaced silently: whatever cannot run a query in the selected backend runs it with
the CUDA_FUSED arithmetic and counts it (`fallback_simt`, `simt_evals`).

### Sine and fast math

Cycles compiles CUDA kernels with `--use_fast_math`, which would turn `sinf`, `expf` and
divisions into approximations. The decoder calls `__nv_sinf`, `__nv_expf` and `__fdiv_rn`
directly, so the `accurate` mode is accurate in every build (fast math still flushes denormals).
`fast` uses `__sinf` on CUDA and `sinf` on the CPU.

## Cycles integration

- **Loader**: `SRNXRImageLoader` (`src/scene/image_vdb.cpp`) dispatches on the format version
  and logs the selected backend, the model summary, warnings and the time mapping.
- **Volume mesh**: the domain box (`is_simple_mesh`), index space = finest octree level.
- **Kernel**: `kernel_image_interp_3d()` (`src/kernel/util/image_3d.h`) calls
  `kernel_image_interp_srnxr_v2()` for format 2 blobs, which is reached from
  `volume_attribute_float4()` (`src/kernel/geom/volume.h`) and from OSL
  (`src/kernel/osl/services_shared.h`). The material maps the fields to density, emission and
  color; nothing optical is in the decoder.
- **Majorants**: Cycles 5.3 builds a 128^3 octree of density bounds by evaluating the volume
  shader at 16 random points per voxel (`kernel_volume_density_evaluate`,
  `src/kernel/bake/bake.h`), which is unsafe for a neural field and costs 33 M network
  evaluations. The bake now sets `SR_VOLUME_MAJORANT_BAKE` (`src/kernel/types.h`, kept by
  `volume_shader_eval` in `src/kernel/integrator/volume_shader.h`); the image then returns the
  upper end (even bake samples) or the lower end (odd samples, `SR_VOLUME_MAJORANT_BAKE_LOWER`)
  of a certified range: the union of the per-leaf output ranges `b +- sum |W|` of the output
  layer (|sin| <= 1) over the leaves whose expanded box (and shutter time interval) touches the
  box of two bake voxels around P. The value at a point is a convex combination of exactly those
  leaves (each of them a neighbor of the strict leaf of the point, so the search runs over the
  neighbor lists of the strict leaves the box touches). This bounds the field; the majorant is
  certified for transfer functions that are monotonic in the field (the test material:
  `max(rho, 0) * scale`). Shaders with a Light Path node make Cycles estimate majorants at render
  time by sampling (not certified); use ray marching (`volume_ray_marching`) for those.
  The certified majorant is loose (on the test model about 9x the mean density, the sum over
  256 neurons of `|W|` assumes every sine at +-1 with the matching sign) and the null-scattering
  integrator evaluates the field at every majorant collision. `srnxr_majorant="sampled"`
  (`SRNXR_DEVICE_FLAG_SAMPLED_MAJORANT`) makes the bake sample the decoded field at the 16 points
  per voxel instead, which is what Cycles does for every procedural density: tight, but NOT
  certified (between the samples the field can exceed it; Cycles then rescales locally), and the
  bake costs 128^3 x 16 queries per image (about 1.7 s on the RTX PRO 6000).
- **Shared decode**: a volume shader reading several fields of one model at one point (density
  and emission of the test scene) decodes once: `kernel_image_interp_srnxr_v2` keeps every field
  of the last decode in the `ShaderData` (`srnxr_cache_*`, reset by `shader_setup_from_volume`),
  keyed by the point and by `SRNXRDeviceHeaderV2::decode_key` (file checksum and decode
  settings, the same for every attribute of one file).
- **Counters**: CUDA kernels keep 16 statistics per image slot in the module global
  `srnxr_counters` (`src/kernel/device/cuda/globals.h`), printed by `CUDADevice` when the device
  is freed; OptiX programs keep them in the blob, printed by `OptiXDevice::mem_free`.
- **Device information**: `DeviceInfo::compute_capability` (CUDA and OptiX devices).
- **COOPVEC**: `Device::supports_neural_coopvec()` (OptiX: `OPTIX_DEVICE_PROPERTY_COOP_VEC` and
  OptiX 9 headers) and `Device::set_use_neural_coopvec()`, called by the XML loader. When set,
  `OptiXDevice::load_kernels` (`src/device/optix/device_impl.cpp`) loads
  `kernel_optix_neural_volume.ptx`, creates the ray generation groups `PG_RGEN_SHADE_VOLUME`,
  `PG_RGEN_SHADE_VOLUME_RAY_MARCHING`, `PG_RGEN_SHADE_SHADOW`, `PG_RGEN_EVAL_VOLUME_DENSITY` and
  the callable groups `PG_CALL_SRNXR_COOPVEC_64/128/256` (placed before the SVM callables so the
  OSL callable indices do not move) in the `PIP_SHADE` pipeline. `OptiXDeviceQueue::enqueue`
  (`src/device/optix/queue.cpp`) launches those four kernels through OptiX and points the
  callable records of these launches at the SRNXR callables. `OptiXDevice::mem_copy_to` converts
  the matrices of an uploaded COOPVEC image with `optixCoopVecMatrixConvert` from the row-major
  staging copy in the blob, after checking that `optixCoopVecMatrixComputeSize` matches the sizes
  the host reserved (the host has no OptiX context and assumes the row-major size); a mismatch is
  a device error. All other kernels stay CUDA kernels.
- **Partially active warps**: the warp-cooperative paths (`srnxr_v2_decode_warp`) use
  `__activemask()`; a fully active warp uses CUDA_TENSOR or the warp SIMT path, 8 - 31 active
  lanes the partial-warp path, fewer lanes one query per thread. Measured on the test scene: 10.9
  active lanes per call on average.
- **Wavefront decode queue** (`volume_neural_queue`, `src/kernel/util/image_3d_srnxr_queue.h`,
  `image_3d_srnxr_queue_decode.h`): the inline decode makes the volume kernels latency bound
  (every thread walks its ray and decodes at every majorant collision, warps are a third active
  at the decoder and the lanes of a nearly empty warp decode one query per thread, 0.6 - 2.3 ms
  each; a launch lasts as long as its longest walk). The queue takes the decodes out of the walk.
  For every `integrator_shade_volume` and `integrator_shade_shadow` launch
  `PathTraceWorkGPU::neural_queue_enqueue` (`src/integrator/path_trace_work_gpu.cpp`) launches:
  1. `integrator_neural_record_volume` / `_shadow`: the same walk with the decoder replaced by
     "append the query to the window of this launch slot, return zero" (`SRNXR_QUEUE_RECORD` in
     `ShaderData::srnxr_queue_mode`, set by `volume_integrate` and
     `integrate_transparent_volume_shadow`). The candidate collisions of null-scattering tracking
     depend on the majorant and the random numbers only, so the recorded queries are a superset
     of the queries of the real walk (which may stop earlier: Russian roulette, zero throughput,
     opaque surfaces). The record walk writes nothing to the film or the path state: the volume
     writes are skipped (`volume_neural_recording`), the shadow walk restores the shadow path
     state it modifies (throughput, volume stack, random number offset, bounces, ray)
     and treats surfaces as transparent (`integrator_shade_shadow_neural_record`).
  2. `neural_queue_expand`: every request is resolved to its leaves (the strict leaf and the
     neighbors whose expanded box contains the point) with the blend exponent, one pair per
     leaf in neighbor order, counted per bucket (`SRNXRDeviceHeaderV2::bucket_base` of the model,
     assigned per decode key by `ImageCache`, plus the network index).
  3. `neural_queue_scan` (one block): pair and tile offsets of the buckets (counting sort).
  4. `neural_queue_scatter`: pair indices sorted by bucket.
  5. `neural_queue_tensor`: the warps of a block take consecutive tiles of 16 rows of one
     network (`srnxr_queue_tensor_group`, `srnxr_v2_eval_network_tensor_block`): the 8 KB
     output tile pairs of the hidden layer weights are staged into shared memory by `cp.async`,
     double buffered, so the layer weights come from L2 once per block instead of once per warp;
     the tiles are full (16 rows) because the rows of a network come from every path of the
     launch, not from one warp.
     One tile per warp (two measured no faster, the fragments spill), the kernel has its own
     launch bounds (256 threads, 255 registers). With `srnxr_tensor_precision="split"`
     (`SRNXR_DEVICE_FLAG_TENSOR_SPLIT`) every activation is an FP16 high part plus the FP16
     rounding of the remainder (`srnxr_split_lo`) and a second mma stream adds the low parts to
     the same FP32 accumulator: about 22 bits of the activation reach the product, the weights
     stay FP16 (the inline CUDA_TENSOR decoder keeps FP16 activations).
  6. `neural_queue_blend`: the pairs of a request are blended in neighbor order (deterministic,
     the same running-max Gaussian blend as the inline decoder) into the result of the request.
  7. the real `integrator_shade_volume` / `integrator_shade_shadow` (`SRNXR_QUEUE_REPLAY`):
     `srnxr_v2_decode_queued` (`src/kernel/util/image_3d.h`) looks every query up in the window
     of its launch slot by exact position, time and decode key (not FIFO: the VSPG finalization
     re-evaluates a visited candidate, the equiangular lookup comes after the loop), and decodes
     inline when the query is not there or its request could not be served (more than
     `SRNXR_QUEUE_MAX_PAIRS` leaves, a network without Tensor Core specialization): correct,
     counted as `replay_miss`.
  The stages are grid-stride kernels over counts in device memory (no host synchronization).
  Statistics (`SRNXR queue stats`: requests, pairs, tiles, overflow, incomplete, replay hits and
  misses, launches) are printed when the path trace work is freed. Memory: `capacity` requests
  (default 4 x the paths) of 16 B position, 12 B ids, 32 B result, 8 pairs of 12 B plus 32 B
  output, and a window of `volume_neural_queue_window` (default 128; 32 overflowed on 0.001 % of the lookups
  of the test scene with the sampled majorant and 0.006 % with the certified one, each miss a
  per-thread inline decode on the critical path) indices per path; about 2.4 GB at 1 M paths,
  10 GB with the 4.6 M paths the RTX PRO 6000 allocates (logged as "Neural volume queue: ...").
  A request the queue cannot hold (capacity) is decoded inline at replay (counted as overflow).
  Limits of this version (reviewed with codex): the queue is configured once per path trace
  work (a changed integrator setting needs a new session); shaders with a Light Path node
  (majorants estimated at render time) or a velocity field read from the model for motion blur
  would make the record walk diverge from the real one (every miss is still decoded inline,
  correctly); the COOPVEC backend (OptiX volume programs) and OSL shading do not replay, the
  queue is disabled for COOPVEC; bucket ranges of unloaded models are reused after 65 536
  networks (a collision only costs inline decodes, the tile kernel checks the model of every
  row). CUDA and OptiX devices only (the stages need `mma.sync`); not with ray marching
  (`volume_ray_marching`) in this version.

## Build options

| CMake option | default | effect |
|---|---|---|
| `WITH_CYCLES_NEURAL_VOLUME` | ON | format 2 loader and decoder (format 1 is always built) |
| `WITH_CYCLES_NEURAL_CUDA_FUSED` | ON | warp-cooperative paths of CUDA_FUSED (without it: one query per thread) |
| `WITH_CYCLES_NEURAL_CUDA_TENSOR` | ON | CUDA_TENSOR (needs `CYCLES_CUDA_BINARIES_ARCH` >= sm_80 for the mma code) |
| `WITH_CYCLES_NEURAL_COOPVEC` | OFF | COOPVEC: OptiX volume programs, needs `OPTIX_ROOT_DIR` with OptiX 9.0+ headers |

The definitions reach the host code, the precompiled CUDA and OptiX kernels and the runtime
kernel compilation (`CUDADevice::compile_kernel_get_common_cflags`). The OptiX 9.1 headers are
public: https://github.com/NVIDIA/optix-dev (tag `v9.1.0`).

## Run time selection

XML attributes of an `<attribute volume_type="srnxr" buffer="file.srnxr" ...>`:

| attribute | meaning |
|---|---|
| `srnxr_backend` | `cuda_fused` (default), `cuda_tensor`, `coopvec` |
| `srnxr_sine` | `accurate` (default), `fast` |
| `srnxr_fields` | one or three field names (default: the only field, or `rho` / `density`) |
| `srnxr_time` | simulation time of the frame (4D models; default: start of the time range, with a warning) |
| `srnxr_time_shutter` | simulation time per shutter, `t = time + shutter * (ray_time - 0.5)` |
| `srnxr_outside` | value outside the domain (default 0) |
| `srnxr_counters` | device statistics on / off (default on) |
| `srnxr_partial_warps` | cooperative decoding of partially active warps on / off (default on) |
| `srnxr_majorant` | majorant bake: `certified` (default, bounds) or `sampled` (decoded field at the bake samples, not certified) |
| `srnxr_tensor_precision` | Tensor Core stage of the wavefront queue: `fp16` (default, activations rounded to FP16 once per layer) or `split` (FP16 high + low part of every activation, two mma streams, FP32-level activations, about 1.3x the tensor stage) |

Integrator sockets: `volume_neural_queue` (off), `volume_neural_queue_window` (128),
`volume_neural_queue_capacity` (0: 4 x the paths).

Command line of `cycles` (overrides the XML): `--neural-queue 0|1`, `--neural-queue-window K`,
`--neural-queue-capacity N`, `--neural-tensor-precision fp16|split`, `--neural-backend`, `--neural-sine`,
`--neural-fallback BACKEND` (used with a warning when the requested backend cannot run on the
render device), `--neural-benchmark` (no fallback, an unsupported backend is an error),
`--neural-counters 0|1`, `--neural-partial-warps 0|1`, `--neural-majorant certified|sampled`. The selected and the requested backend
are logged for every attribute.
Support rules: CPU runs CUDA_FUSED; CUDA_TENSOR needs a CUDA or OptiX device with SM 8.0+ and SVM
shading; COOPVEC needs the OptiX device, SVM shading, the build option and the device property.
A backend change needs a scene reload (the backend is part of the device blob).

## Reproduction

On p12 (paths of this work; nothing is installed in the home directory):

```sh
# models (local machine with torch, or any machine with numpy for --fit lstsq)
python tools/srnxr/srnxr_make_synthetic.py synth_w256_l3.srnxr --width 256 --layers 3 --threshold 0.5 --steps 3000
python tools/srnxr/srnxr_make_synthetic.py synth_w256_l4.srnxr --width 256 --layers 4 --threshold 0.5 --steps 3000
python tools/srnxr/srnxr_points.py synth_w256_l3.srnxr points/l3
python tools/srnxr/srnxr_analysis.py points synth_w256_l3.srnxr points/l3

# standalone tools (CUDA 12.9, OptiX 9.1 headers for srnxr_coopvec)
cmake -S tools/srnxr/bench -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DSRNXR_OPTIX_ROOT=<optix-dev-9.1.0>
cmake --build build
build/srnxr_bench synth_w256_l3.srnxr points/l3_accuracy.f32 --backend cuda_tensor --out tensor.f32
build/srnxr_bench_cycles_regs synth_w256_l3.srnxr points/l3_coherent.f32 --backend cuda_tensor --block 384 --batches 1,32,256,1024,4096,16384,65536
build/srnxr_coopvec synth_w256_l3.srnxr points/l3_accuracy.f32 --out coopvec.f32
python tools/srnxr/srnxr_compare.py synth_w256_l3.srnxr points/l3_accuracy.f32 tensor=tensor.f32 coopvec=coopvec.f32
python tools/srnxr/srnxr_bounds_check.py synth_w256_l3.srnxr

# Cycles
python tools/srnxr/srnxr_make_scene.py synth_w256_l3.srnxr scene.xml --time 0.5
# --width/--height: the standalone renders 1024 x 512 otherwise, whatever the XML camera says
cycles --background --device CUDA --neural-backend cuda_tensor --width 960 --height 540 --samples 16 --output out.png scene.xml
# COOPVEC: Cycles configured with -DWITH_CYCLES_NEURAL_COOPVEC=ON -DOPTIX_ROOT_DIR=<optix-dev-9.1.0>
cycles --background --device OPTIX --neural-backend coopvec --width 960 --height 540 --samples 16 --output out.png scene.xml
```

Cluster scripts: `/mnt/proj1/open-36-34/milanjaros/projects/blender/scripts/srnxr/`
(`env.sh`, `build_cycles.sh`, `run_accuracy.sh`, `run_perf.sh`, `run_render.sh`).

## Limitations

- Synthetic models only; no real checkpoint of the poster's surrogate was available.
- Fourier MLP models, time-partitioned leaves and a sine on the output layer are rejected.
- Several fields of one file read by several attributes are stored separately (one device blob
  per attribute); the decode of a point is shared (see Shared decode).
- CUDA_TENSOR needs fully active warps; in the Cycles volume integrator most warps are partial
  (10.9 of 32 lanes on the test scene), so only a quarter of the evaluations reach the Tensor
  Cores and the end-to-end time equals CUDA_FUSED.
- Use named voxel attributes (`std="ATTR_STD_NONE"`) for SRNXR fields read by attribute nodes:
  an attribute node reading "density" did not reliably find an attribute declared with
  `std="ATTR_STD_VOLUME_DENSITY"` in the XML loader.
- The majorant bound is certified for monotonic transfer functions only, and is not used for
  shaders with a Light Path node (Cycles samples those at render time).
- COOPVEC inside Cycles needs the OptiX device, SVM shading, OptiX 9 headers and a background
  render (the backend has to be set before the kernels are loaded); the CUDA kernels (and the CPU)
  never run cooperative vectors. The first compile of its OptiX module takes about 3 minutes.
- Statistics are not kept on the CPU.
- Untested: RTX 4090 / SM 8.9 and every other GPU than the RTX PRO 6000 Blackwell, HIP, Metal,
  oneAPI (Metal does not build format 2).
