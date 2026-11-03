# Neural volume decoder: results

Measured results of the SRNXR format 2 decoder (`tools/srnxr/neural_volume.md`): standalone
throughput and accuracy of the three inference backends, and the first end-to-end Cycles renders.
Every number below was measured; combinations that were not run are marked **NOT TESTED**.

> **All models are SYNTHETIC.** No trained checkpoints of the adaptive-octree CFD surrogate
> (Smolík, Brzobohatý, HPCSE 2026) were available. The models are SIRENs trained by
> `tools/srnxr/srnxr_make_synthetic.py` on an analytic time-dependent flow. They have the
> layout of the method (adaptive octree, 15 % expanded leaves, Gaussian blending, one SIREN per
> leaf over (x, y, z, t), five fields `ux uy uz p rho`), but say nothing about the fidelity of
> real models. Accuracy limits for real data must come from the dataset requirements.

## Hardware and software

| item | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (2 per node), SM 12.0, 97 887 MiB |
| node | p12-turin01 (IT4I Complementary Systems) |
| driver | 610.43.02 (CUDA driver API 13.3) |
| CUDA toolkit | 12.9.0 (runtime 12.9), GCC 14.3 |
| OptiX | Cycles build: SDK 8.1.0; `srnxr_coopvec`: OptiX 9.1.0 headers (github.com/NVIDIA/optix-dev, tag v9.1.0) |
| clocks | max SM clock 2430 MHz, power limit 600 W, application clocks not locked |
| Cycles | branch `srnxr_tensor` at 3cf99ca8e plus the uncommitted changes of this work |
| kernel flags | standalone: `-O3 -lineinfo`, `sm_120`, no fast math; `_cycles_regs` variant adds `-maxrregcount=168` and runs 384-thread blocks like the Cycles kernels; Cycles: `--use_fast_math` (the decoder calls `__nv_sinf`, `__nv_expf`, `__fdiv_rn` explicitly in accurate mode) |
| RTX 4090 (SM 8.9) | **NOT TESTED** (not available) |

## Models

| model | leaves | octree depth | network | FP16 weights | file (with FP32 master) |
|---|---|---|---|---|---|
| SIREN-3 (`synth_w256_l3`) | 120 | 3 | 4 -> 256 x 3 sine layers -> 5, w0 = 30 every layer | 32.0 MB | 96.4 MB |
| SIREN-4 (`synth_w256_l4`) | 120 | 3 | 4 -> 256 x 4 sine layers -> 5, w0 = 30 every layer | 47.7 MB | 143.7 MB |

Fit of the synthetic models against the analytic field (held-out NRMSE, range normalized):
SIREN-3 0.08 % (rho) - 0.44 % (uz); SIREN-4 0.08 % - 0.44 %. Leaves overlap 15.0 % of the strict
size on every interior side (measured by the loader), 18.2 neighbors per leaf on average, 27 at
most.

## Standalone throughput

`tools/srnxr/bench/srnxr_bench` and `srnxr_coopvec`, decoding the Cycles device blob with the
Cycles decoder code. Mqueries/s at batch 65 536 and the latency of a batch of 1 query (one
launch, launch overhead included; an empty launch costs 1.45 us). Warm runs, median of 20.
A query is one field evaluation, which evaluates 1 - 8 local networks (the leaves whose support
contains the point).

Workloads (65 536 queries each, `tools/srnxr/srnxr_points.py`):
- `one_leaf`: all queries in the strict interior of one leaf (one network per query, best case)
- `coherent`: 32-query segments along random rays (neighboring queries share leaves)
- `random`: uniform random points (neighboring queries in different leaves)
- `overlap`: points in the overlap zones only (several networks per query)

### SIREN-3, Cycles launch configuration (168 registers, 384 threads per block)

| workload | CUDA_FUSED per-thread | CUDA_FUSED warp | CUDA_FUSED warp, `__sinf` | CUDA_TENSOR | CUDA_TENSOR, `__sinf` | COOPVEC (OptiX) | COOPVEC, `__sinf` |
|---|---|---|---|---|---|---|---|
| one_leaf | 2.13 Mq/s, 1649 us@1 | 22.4 Mq/s, 144 us@1 | 22.9 Mq/s, 142 us@1 | **51.7 Mq/s**, 94 us@1 | **59.4 Mq/s**, 80 us@1 | 3.40 Mq/s, 645 us@1 | 3.43 Mq/s, 621 us@1 |
| coherent | 1.71 Mq/s, 3287 us@1 | 8.38 Mq/s, 273 us@1 | 8.59 Mq/s, 271 us@1 | **45.6 Mq/s**, 176 us@1 | **52.6 Mq/s**, 146 us@1 | 3.13 Mq/s, 1284 us@1 | 3.21 Mq/s, 1226 us@1 |
| random | 1.11 Mq/s, 3285 us@1 | 8.31 Mq/s, 275 us@1 | 8.55 Mq/s, 272 us@1 | 11.0 Mq/s, 178 us@1 | 11.6 Mq/s, 148 us@1 | 1.29 Mq/s, 1284 us@1 | 1.31 Mq/s, 1227 us@1 |
| overlap | 0.68 Mq/s, 6560 us@1 | 4.84 Mq/s, 535 us@1 | 4.95 Mq/s, 527 us@1 | 5.48 Mq/s, 340 us@1 | 5.66 Mq/s, 281 us@1 | 0.86 Mq/s, 2414 us@1 | 0.86 Mq/s, 2316 us@1 |

COOPVEC runs as an OptiX raygen program (OptiX chooses the launch configuration), not with the
Cycles launch bounds.

### SIREN-3, unconstrained registers (202 registers, 128 threads per block)

| workload | CUDA_FUSED per-thread | CUDA_FUSED warp | CUDA_FUSED warp, `__sinf` | CUDA_TENSOR | CUDA_TENSOR, `__sinf` |
|---|---|---|---|---|---|
| one_leaf | 1.81 Mq/s | 19.8 Mq/s | 20.2 Mq/s | 34.4 Mq/s | 39.6 Mq/s |
| coherent | 1.45 Mq/s | 9.72 Mq/s | 10.6 Mq/s | 37.6 Mq/s | 43.9 Mq/s |
| random | 0.85 Mq/s | 7.98 Mq/s | 8.30 Mq/s | 8.10 Mq/s | 9.42 Mq/s |
| overlap | 0.55 Mq/s | 4.40 Mq/s | 4.49 Mq/s | 4.16 Mq/s | 4.79 Mq/s |

The 168-register / 384-thread configuration is faster for CUDA_TENSOR (more resident warps hide
the L2 latency of the weight fragments) despite 184 - 248 bytes of spills.

### SIREN-4, Cycles launch configuration (168 registers, 384 threads per block)

| workload | CUDA_FUSED per-thread | CUDA_FUSED warp | CUDA_FUSED warp, `__sinf` | CUDA_TENSOR | CUDA_TENSOR, `__sinf` | COOPVEC (OptiX) | COOPVEC, `__sinf` |
|---|---|---|---|---|---|---|---|
| one_leaf | 1.41 Mq/s | 14.9 Mq/s | 15.3 Mq/s | **39.6 Mq/s** | **46.4 Mq/s** | 2.42 Mq/s, 891 us@1 | 2.44 Mq/s, 866 us@1 |
| coherent | 1.05 Mq/s | 5.59 Mq/s | 5.73 Mq/s | **35.2 Mq/s** | **41.9 Mq/s** | 2.23 Mq/s, 1816 us@1 | 2.28 Mq/s, 1751 us@1 |
| random | 0.73 Mq/s | 5.86 Mq/s | 6.00 Mq/s | 8.65 Mq/s | 9.38 Mq/s | 0.89 Mq/s, 1714 us@1 | 0.89 Mq/s, 1654 us@1 |
| overlap | 0.46 Mq/s | 3.39 Mq/s | 3.47 Mq/s | 4.33 Mq/s | 4.62 Mq/s | 0.58 Mq/s, 3421 us@1 | 0.59 Mq/s, 3287 us@1 |

### SIREN-4, unconstrained registers (128 threads per block)

| workload | CUDA_FUSED per-thread | CUDA_FUSED warp | CUDA_FUSED warp, `__sinf` | CUDA_TENSOR | CUDA_TENSOR, `__sinf` |
|---|---|---|---|---|---|
| one_leaf | 1.20 Mq/s | 14.6 Mq/s | 15.2 Mq/s | 29.0 Mq/s | 33.6 Mq/s |
| coherent | 1.00 Mq/s | 7.50 Mq/s | 7.84 Mq/s | 31.1 Mq/s | 36.7 Mq/s |
| random | 0.58 Mq/s | 5.76 Mq/s | 5.95 Mq/s | 6.44 Mq/s | 7.53 Mq/s |
| overlap | 0.38 Mq/s | 3.15 Mq/s | 3.24 Mq/s | 3.24 Mq/s | 3.79 Mq/s |

The fourth sine layer costs CUDA_TENSOR 15 - 23 % and the warp SIMT path 25 - 35 %, in line with
the extra 256 x 256 layer (about +50 % multiply-adds).

### Batch size scaling

All batch sizes 1, 32, 256, 1024, 4096, 16 384 and 65 536 are in
`neural_volume_results/perf.csv` (one row per build, model, workload, configuration and batch).

### Execution evidence

Device counters of the correctness runs (16 000 accuracy points, 33 816 network evaluations):

| backend | counters |
|---|---|
| CUDA_FUSED per-thread | 33 816 `simt_evals` |
| CUDA_FUSED warp | 33 816 `warp_simt_evals` in 21 539 passes, 0 per-thread |
| CUDA_TENSOR | 20 878 `tensor_tiles` with 33 816 rows (1.6 of 16 rows per tile on this incoherent set), 0 `fallback_simt` |
| COOPVEC | 33 816 `coopvec_evals`, 0 `simt_evals` |

`OPTIX_DEVICE_PROPERTY_COOP_VEC` reports `STANDARD` on this GPU and driver. The OptiX module
with FP16 cooperative vector products compiles and runs; the same products with **FP32 output
vectors are rejected by OptiX 9.1** at module creation (`output vector type is of unsupported
type (0x2A03)`), so COOPVEC rounds every layer output (the sine argument) to FP16.

### Why the backends perform as they do

- **CUDA_FUSED per-thread** keeps two 256-float activation arrays per thread in local memory
  (2 KB per thread): L1 thrashing and long dependent load chains. It remains as the CPU path and
  as the GPU path for warps with fewer than 8 active lanes. (The per-thread numbers of the tables
  above are of the first version, which summed one neuron at a time; it now sums 8 rows at once
  with bitwise identical results: 6.11 Mq/s and 586 us@1 on one_leaf, 3.21 Mq/s on random,
  2.9 - 3.0x, see Interactive rendering.)
- **CUDA_FUSED warp-cooperative** distributes the 256 neurons over the lanes (8 per lane),
  broadcasts the inputs by shuffle and reads the weights from a transposed FP16 copy, coalesced;
  4 queries per pass. 10x the per-thread path at identical arithmetic.
- **CUDA_TENSOR** groups the queries of a warp by network (16 rows per `mma.m16n8k16` tile) and
  keeps the activations in registers in fragment layout. It wins when neighboring queries share
  networks (one_leaf, coherent: 45 - 59 Mq/s); with random leaves the tiles are 1 - 2 rows full
  and it falls to the level of the warp SIMT path.
- **COOPVEC** is 10 - 15x slower than CUDA_TENSOR here: OptiX has no cooperative vector sine, so
  the 256-element vectors of every layer are taken apart for a per-element FP32 sine, and the
  256-wide FP16 vectors per thread do not fit the register file. The `__sinf` variant is only 1 %
  faster, the sine is not the bottleneck.
- Fast sine (`__sinf`) gains 2 - 15 % on CUDA_FUSED and CUDA_TENSOR.
- Network divergence (`random`, `overlap`) costs every backend 4 - 10x; reordering queries by
  network before decoding is not implemented (and is not possible inline in the Cycles volume
  integrator).

### Partially active warps

In the Cycles volume integrator most warps reach the image lookup with only part of their lanes
active (on the test scene 10.9 of 32 on average, see below). `srnxr_bench --active-fraction F`
reproduces this: every launch is split into two launches over complementary random subsets of
the threads (F and 1 - F active). CUDA_FUSED, SIREN-3, Cycles launch configuration, batch
65 536, Mqueries/s:

| workload | F | partial-warp path | per-thread (`--no-partial`) | gain |
|---|---|---|---|---|
| one_leaf | 0.7 / 0.5 / 0.3 | 9.83 / 7.60 / 8.96 | 1.22 / 1.23 / 1.26 | 6.2 - 8.1x |
| coherent | 0.7 / 0.5 / 0.3 | 3.51 / 2.90 / 3.47 | 0.85 / 0.79 / 0.79 | 3.7 - 4.4x |
| random | 0.7 / 0.5 / 0.3 | 5.41 / 3.85 / 5.48 | 0.72 / 0.71 / 0.74 | 5.4 - 7.6x |
| overlap | 0.7 / 0.5 / 0.3 | 2.88 / 2.24 / 3.04 | 0.45 / 0.46 / 0.46 | 4.9 - 6.7x |

The partial-warp path matches the per-thread path to 2.4e-7 (97.8 % of the values bitwise; the
remaining differences are FMA contraction); with F = 0.5 CUDA_TENSOR runs all evaluations through
it and counts them as `fallback_simt`. It is limited by re-reading the weights for every pass of
2 queries (the static shared memory of a Cycles block does not hold more FP32 activations).

## Load, packing and memory (SIREN-3)

| step | value |
|---|---|
| parse + validate (host) | 0.42 s (96 MB file, CRC of every section) |
| pack device blob (host) | 0.025 s |
| device blob, CUDA_FUSED | 64.4 MB (FP16 weights 32.4 MB + transposed warp copy 32.0 MB) |
| device blob, CUDA_TENSOR | 64.4 MB (FP16 weights 32.4 MB + mma fragment copy 31.9 MB) measured before the partial-warp path; it now also holds the transposed copy (+32.0 MB) |
| device blob, COOPVEC | weights + 32.3 MB cooperative vector region (INFERENCING_OPTIMAL matrices = row-major size); in Cycles also the 32.3 MB row-major staging copy the device converts from |
| `optixCoopVecMatrixConvert`, 360 matrices | 2.1 - 2.2 ms standalone (6.4 ms the first time), 1.8 - 2.1 ms in Cycles |
| OptiX module + pipeline creation, standalone | 10.5 ms (cached) |
| OptiX neural volume module in Cycles, first compile | about 196 s, 10 GB host memory (cached afterwards); with the network evaluation inlined into every call site it did not finish in 15 minutes (60 GB) |

In Cycles every SRNXR attribute is its own image, so a scene that reads two fields of one file
(density and pressure) holds two device blobs.

## Accuracy

`tools/srnxr/srnxr_compare.py` against the float64 oracle with the deployed FP16 weights
(`srnxr_reference.py`), 16 000 deterministic points in nine categories (block centers, strict
boundaries, overlap zones, T-junctions, high-gradient front, time boundaries, outside the domain,
uniform). NRMSE is RMSE / field range over all points inside the domain; PSNR over the range.

| comparison (SIREN-3) | NRMSE ux | NRMSE p | NRMSE rho | max abs p | PSNR p |
|---|---|---|---|---|---|
| storage: FP16 weights vs FP32 master (oracle arithmetic) | 3.5e-5 | 4.2e-5 | 2.8e-5 | 1.0e-3 | 87.6 dB |
| CPU (host build of CUDA_FUSED) | 5.0e-8 | 5.7e-8 | 4.0e-8 | 1.1e-6 | 144.9 dB |
| CUDA_FUSED per-thread | 4.9e-8 | 5.7e-8 | 4.1e-8 | 1.1e-6 | 144.9 dB |
| CUDA_FUSED warp | 4.2e-8 | 4.9e-8 | 3.5e-8 | 9.2e-7 | 146.3 dB |
| CUDA_FUSED warp, `__sinf` | 6.5e-8 | 7.5e-8 | 5.7e-8 | 1.3e-6 | 142.6 dB |
| CUDA_TENSOR | 1.4e-5 | 1.6e-5 | 1.2e-5 | 2.8e-4 | 96.1 dB |
| CUDA_TENSOR, `__sinf` | 1.3e-5 | 1.6e-5 | 1.2e-5 | 2.8e-4 | 96.1 dB |
| emulation: FP16 activations per layer (oracle) | 1.3e-5 | 1.6e-5 | 1.2e-5 | 2.8e-4 | 96.1 dB |
| COOPVEC (FP16 layer outputs) | 9.4e-5 | 1.1e-4 | 8.4e-5 | 2.3e-3 | 79.1 dB |
| COOPVEC, `__sinf` | 9.3e-5 | 1.1e-4 | 8.4e-5 | 2.3e-3 | 79.1 dB |

SIREN-4 behaves the same (worst field: FUSED 6.8e-8, TENSOR 2.3e-5, COOPVEC 1.7e-4, storage 6.3e-5).
All backends return exactly the outside value for the 2 000 points outside the domain.

- CUDA_TENSOR matches the oracle with FP16-rounded activations, so its error is the intended
  activation rounding, below the FP16 storage error.
- COOPVEC is 7x less accurate than CUDA_TENSOR and above the storage error: the pre-activation
  is rounded to FP16 before the multiplication by w0 = 30, which amplifies it.
- The fast sine adds nothing measurable on top of FP16 effects; on CUDA_FUSED it is 1.5x the
  FP32 rounding error.

Field-level analysis (`srnxr_analysis.py`, slice z = -0.3, t = 0.5, 256 x 256, SIREN-3):

| backend | gradient rel. RMSE (rho) | rho isosurface displacement, mean | support jump rho, mean / max |
|---|---|---|---|
| oracle | 0 | 0 | 1.6e-4 / 3.3e-3 |
| CPU | 7.5e-6 | 3.3e-8 | 1.6e-4 / 3.3e-3 |
| CUDA_FUSED warp | 6.5e-6 | 3.0e-8 | 1.6e-4 / 3.3e-3 |
| CUDA_TENSOR | 2.5e-3 | 9.9e-6 | 1.7e-4 / 3.3e-3 |
| COOPVEC | 1.4e-2 | 5.3e-5 | 2.0e-4 / 3.3e-3 |

The isosurface displacement is in domain units (grid spacing 7.8e-3). The jump across the support
boundary of a leaf is a property of the method (Gaussian weights truncated at the expanded box),
not of the backends. Error maps: `neural_volume_results/error_map_*.png`.

Certified majorant bounds (`srnxr_bounds_check.py`, 2 000 samples per leaf): 0 violations of the
per-leaf output ranges and of the blended field against the neighbor union; the bound is 1.4 - 2.0x
the sampled range (median), up to 13x for rho in nearly constant leaves. As a density majorant on
a 24^3 voxel grid (`srnxr_majorant_check.py`): 9.0x the mean density with the bake of the
contributing leaves, 18.2x with the earlier neighbor union, 1.06x for the sampled maximum.

## Cycles end to end

> Measured **before** the majorant, shared decode and per-thread changes described in
> [Interactive rendering](#interactive-rendering), which make the frames 4.4x (certified majorant)
> to 13x (sampled majorant) faster; the 16-sample times below were not re-measured.

Scene `srnxr_make_scene.py`: 1024 x 512 (the default output size of the `cycles` standalone,
which does not take the camera resolution of the XML unless `--width`/`--height` are given), the
SIREN-3 volume (attribute `neural_density` from
field `rho`, `pressure` from field `p`; density `max(rho, 0) * 8`, emission `max(p, 0) * 0.6`),
one point light, 16 samples, default (unbiased) volume integrator, one Cycles build with OptiX 9.1
headers and `WITH_CYCLES_NEURAL_COOPVEC=ON` (`install/cyclesphi_srnxr_coopvec_p12`). Every row
decodes both fields (two images, about 87 M queries and 156 M network evaluations each).

| device, backend | partial-warp path | render time | decoder counters (both images) |
|---|---|---|---|
| CUDA, CUDA_FUSED | on | **94.5 s** | warp-cooperative 81.0 M, partial-warp 189.4 M, per-thread 41.8 M |
| CUDA, CUDA_FUSED | off (`--neural-partial-warps 0`) | 211.6 s | warp-cooperative 80.9 M, per-thread 231.3 M |
| CUDA, CUDA_TENSOR | on | **94.4 s** | mma 12.6 M tiles / 81.2 M rows (6.4 - 6.6 of 16 rows per tile); `fallback_simt` 231.0 M (partial-warp 189.0 M, per-thread 42.0 M) |
| CUDA, CUDA_TENSOR | off | 210.0 s | mma 12.5 M tiles / 80.9 M rows; `fallback_simt` 231.3 M, all per-thread |
| OptiX, CUDA_TENSOR | on | 92.5 s | as on the CUDA device |
| OptiX, COOPVEC | - | 138.1 s | 312.2 M `coopvec_evals`, 0 SIMT |
| CPU | | NOT TESTED | |

Images: `neural_volume_results/render_l3_{cuda_fused,cuda_tensor,optix_coopvec}_16spp.png`.
PSNR against CUDA_FUSED: CUDA_FUSED without the partial-warp path 71.1 dB, CUDA_TENSOR 48.0 dB,
COOPVEC 44.6 dB (the backends change the decoded values slightly, so the paths and the noise
differ); mean image values agree to 5 digits.

- The warps are partially active at the image lookup: 85.9 M queries in 7.85 M warp calls,
  10.9 active lanes on average. The partial-warp path halves the render time for both CUDA
  backends (211 s -> 94 s); 13 % of the evaluations still come from warps with fewer than 8
  active lanes and run per thread.
- CUDA_TENSOR runs only 26 % of the evaluations on Tensor Cores (the fully active warps) and is
  no faster than CUDA_FUSED end to end: in this integrator the warp occupancy, not the matrix
  arithmetic, is the limit.
- COOPVEC runs every evaluation as cooperative vector products inside the OptiX volume programs
  and is 1.46x slower than the CUDA backends, in line with the standalone measurements.
- Every run includes the conservative majorant bake: 33.6 M bound queries per image (128^3 voxels
  x 16 samples), no network evaluation.

An earlier version of the test scene declared the density attribute with
`std="ATTR_STD_VOLUME_DENSITY"`; the attribute node did not reliably find it (depending on the
image load order), so the earlier timings of this section (335 s, 143 s) decoded only one field
and are withdrawn. The scene now uses named attributes.

## Interactive rendering

Progressive rendering at one sample per pixel per frame, the mode of interactive (viewport)
rendering: the frame time is the path tracing time of one sample. Measured with the `cycles`
standalone in background mode (script `scripts/srnxr/run_fps.sh` in the project directory on p12;
data in `neural_volume_results/fps.csv`, the first measurement in `fps_v1.csv`): render time with
9 samples minus render time with 1 sample, divided by 8, so scene loading, kernel loading and the
majorant bake are excluded; display transfer and denoising are not included (no interactive
display on the node). GPU: RTX PRO 6000 Blackwell, counters off, partial-warp path on unless
stated, test scene of `srnxr_make_scene.py` with `--width`/`--height` passed to `cycles`.
"both" decodes density and pressure, "density" only the density.

| resolution | scene | first measurement | CUDA_FUSED certified | CUDA_TENSOR certified | COOPVEC certified | CUDA_FUSED sampled | CUDA_TENSOR sampled | COOPVEC sampled |
|---|---|---|---|---|---|---|---|---|
| 480 x 270 | density | 1768 ms, 0.57 FPS | 629 ms, 1.59 FPS | 603 ms, 1.66 FPS | 1421 ms, 0.70 FPS | 222 ms, **4.50 FPS** | 231 ms, 4.32 FPS | 463 ms, 2.16 FPS |
| 480 x 270 | both | 3098 ms, 0.32 FPS | 741 ms, 1.35 FPS | 739 ms, 1.35 FPS | 1368 ms, 0.73 FPS | 266 ms, 3.76 FPS | 259 ms, **3.87 FPS** | 429 ms, 2.33 FPS |
| 960 x 540 | density | 4043 ms, 0.25 FPS | 1409 ms, 0.71 FPS | 1366 ms, 0.73 FPS | 4020 ms, 0.25 FPS | 456 ms, 2.19 FPS | 446 ms, **2.24 FPS** | 1163 ms, 0.86 FPS |
| 960 x 540 | both | 6814 ms, 0.15 FPS | 1550 ms, 0.65 FPS | 1553 ms, 0.64 FPS | 3698 ms, 0.27 FPS | 504 ms, **1.99 FPS** | 510 ms, 1.96 FPS | 1097 ms, 0.91 FPS |
| 1920 x 1080 | density | 12486 ms, 0.080 FPS | 4407 ms, 0.23 FPS | 4315 ms, 0.23 FPS | 14656 ms, 0.068 FPS | 1399 ms, 0.71 FPS | 1393 ms, **0.72 FPS** | 4183 ms, 0.24 FPS |
| 1920 x 1080 | both | 21440 ms, 0.047 FPS | 4874 ms, 0.21 FPS | 4772 ms, 0.21 FPS | 13342 ms, 0.075 FPS | 1560 ms, 0.64 FPS | 1545 ms, **0.65 FPS** | 3930 ms, 0.25 FPS |

"first measurement" is CUDA_FUSED before the three changes below; without the partial-warp path
(`--neural-partial-warps 0`), 960 x 540, both fields, CUDA_FUSED: 15331 ms then, 2395 ms
(certified) and 724 ms (sampled) now.

What the first measurement got wrong, found with per-kernel timings
(`CYCLES_DEBUG_PER_KERNEL_PERFORMANCE=1 cycles --log-level trace`, script
`scripts/srnxr/run_kstats.sh`): the whole frame is `integrator_shade_volume` and
`integrator_shade_shadow` (the majorant bake takes 11 - 41 ms once), and those kernels were
latency bound, not throughput bound: at 960 x 540 a launch with 103 K paths took longer (0.84 s)
than the one with 232 K paths (0.80 s), and a launch with 29 K paths still 0.38 s. Every thread
walks its whole ray segment through the volume in one launch, one field decode per majorant
collision, and the launch lasts as long as its longest walk.

1. **Loose majorant.** The bake took the union of the leaf ranges over the neighbors of every
   leaf near the bake sample (neighbors of neighbors): about 18x the mean density on a voxel
   grid of the test model (`tools/srnxr/srnxr_majorant_check.py`), so about 95 % of the collisions were null
   collisions, each with a decode. The bake now takes the union over the leaves whose expanded
   box touches the voxel, of which the field is a convex combination (still certified, about
   9x). The lower or upper end of the bounds was also chosen by the shader LCG, so a voxel had a
   2^-16 chance of never seeing the upper end; the bake samples now alternate deterministically.
   960 x 540, density, 1 sample: 4.05 s -> 3.30 s.
2. **Separate decodes per field.** Density and emission read two fields of one model at the same
   point and decoded twice; the decode is now shared through the shader data (both fields cost
   1.1x one, before 1.7x).
3. **Slow per-thread path.** Warps with fewer than 8 active lanes, which are the long tails of the
   launches, decode per thread. That path summed one neuron at a time (a 256-long dependent FMA
   chain, the activations re-read from local memory for every neuron); it now sums 8 rows at once
   (`srnxr_dot_rows`, each row still in index order, so the results are bitwise identical:
   0 of 320 000 bytes differ on the accuracy set). Standalone, Cycles launch configuration:
   latency of one query 1739 -> 586 us (one_leaf) and 6923 -> 2317 us (overlap), throughput
   2.14 -> 6.11 Mq/s (one_leaf), 1.05 -> 3.21 Mq/s (random). 960 x 540, density, 1 sample:
   3.30 s -> 1.41 s.

`--neural-majorant sampled` (`srnxr_majorant="sampled"`) bakes the majorant from the decoded field
at the 16 bake samples per voxel, which is what Cycles does for every procedural density: about
1.06x the mean density, another 3x faster, but **not certified** (between the samples the field
can exceed it; Cycles then rescales the majorant locally) and the bake costs 128^3 x 16 decodes per
image (1.6 - 1.7 s at scene load, 35 M network evaluations/s). The specification forbids a
majorant from sampling alone, so the default stays `certified`. On the test scene the two agree:
480 x 270, both fields, 256 samples, the mean image values differ by 1.1e-5, the largest 16 x 16
block mean difference is 0.0017 (below the 8-bit step) and the PSNR between them is 45.9 dB
(noise).

- With the sampled majorant one sample of 960 x 540 decodes only about 0.7 M points, 1.4 M network
  evaluations (counters; the remaining 33.5 M queries of the run are the bake), so 450 ms per frame
  is about 3 M network evaluations/s: the frame time is still the latency of the longest sequential walks, while the
  same GPU decodes 35 M networks/s in the coherent bake.
- CUDA_TENSOR does not help end to end (partially active warps). COOPVEC is now 2 - 3x slower than
  the CUDA backends: it has no per-thread path to improve (its products are per thread already)
  and it rounds every layer to FP16.
- Remaining levers, not implemented: a smaller network (one decode is 4 -> 256 x 3 -> 5, about
  240 K multiply-adds per query with the blending, versus about 2 K for the 32 - 64 wide MLPs
  over a latent feature grid of the neural shading samples, github.com/shader-slang/neural-shading-s26,
  which also use activations cooperative vectors evaluate natively; the sine is not one of them);
  shorter dependent walks (reordering or batching the decodes of many paths between launches,
  which the Cycles volume integrator does not allow); a tighter certified bound than
  `b +- sum |W|`.
- Not measured: the GUI viewport of `cycles` / cyclesphi with display and denoising, adaptive
  resolution (start pixel size) and sample count of the interactive session.

## Wavefront decode queue (2026-11-03)

The frame time of the interactive renders above is the latency of the longest per-thread walks
(0.5 s per sample at 960 x 540 for 0.7 M decodes: about 3 % of the decoder throughput of the
GPU). The wavefront queue (`tools/srnxr/neural_volume.md`, "Wavefront decode queue") records
the queries of the walks, decodes them in batches sorted by network with full Tensor Core tiles,
and replays them. Standalone measurement of the batch stages
(`srnxr_bench --wavefront`, build with nvcc 13.3 `-O3 -lineinfo`, `sm_120`, 256 threads per
block without a register limit; the point sets of `srnxr_points.py` with `--workload-size
1048576`), RTX PRO 6000 Blackwell, SIREN-3, accurate sine. Every point is one request; the
stages record, expand, scan, scatter, tensor, blend and replay run as separate kernels and are
timed with events (median of 5 - 10). Mpairs/s counts the local network evaluations.

| workload | requests | pairs per request | direct CUDA_TENSOR decode | queue, block-shared weights | queue, independent warp tiles | tensor stage (block) |
|---|---|---|---|---|---|---|
| coherent | 65 536 | 1.75 | 32.7 Mq/s | **100.6 Mq/s** (0.65 ms) | 99.3 Mq/s | 0.49 ms |
| random | 65 536 | 1.77 | 7.1 Mq/s | **98.2 Mq/s** (0.67 ms) | 103.9 Mq/s | 0.54 ms |
| coherent | 1 048 576 | 1.76 | 56.4 Mq/s | **117.4 Mq/s, 206.8 Mpairs/s** (8.9 ms) | 103.9 Mq/s | 6.8 ms |
| random | 1 048 576 | 1.77 | 10.2 Mq/s | **120.7 Mq/s, 213.3 Mpairs/s** (8.7 ms) | 110.6 Mq/s | 7.0 ms |
| overlap | 1 048 576 | 3.62 | 5.4 Mq/s | **67.4 Mq/s, 243.8 Mpairs/s** (15.6 ms) | 54.1 Mq/s | 13.3 ms |

- The tiles are full: 15.9 - 16.0 rows per tile on every workload (the direct decoder of the
  interactive renders had 6.4 - 6.6 rows, and 1 - 2 rows on `random`); the network order of
  the queries no longer matters (`random` decodes as fast as `coherent`).
- The stages other than the tensor stage cost about 1.9 ms per 1 M requests (expand 0.8 - 1.1
  ms, scatter 0.6 - 0.9 ms, blend 0.1 - 0.2 ms, record and replay 0.1 ms each, scan 4 us).
- Correctness: 0 replay misses, 0 overflow, 0 incomplete requests; the blended fields differ
  from the direct CUDA_TENSOR decode by at most 3.6e-7 (1.2e-7 of the field range; the blend
  order of the contributing leaves differs).
- Block-shared weights (12 warps of a block evaluating tiles of one network, 8 KB weight panels
  staged in shared memory with `cp.async`, double buffered) are 13 - 25 % faster than
  independent warp tiles at 1 M requests and equal at 65 K requests. A microbenchmark of
  `mma.sync.m16n8k16` (`scratch mma_bench.cu`, 752 warps x 4 independent chains) measures
  447 TFLOPS dense FP16 on this GPU with FP32 accumulation and 445 with FP16 accumulation
  (16.8 cycles per mma per SM sub-partition at 2430 MHz; the FP32 accumulate is not halved on the
  RTX PRO 6000): the 1024 mma of a tile (two 256 x 256 layers) would take 1.8 us per tile and SM,
  the tensor stage takes 9.8 us (6.9 ms for 115 498 tiles on 188 SMs), 18 % of the Tensor Core
  peak. The fast sine (`--sine fast`) lowers it to 8.6 us (5.9 ms), so the accurate sine is 14 %
  of the stage; the rest is the issue and latency of the dependent mma chains, the fragment loads,
  the conversions and the block barriers. Nsight Compute is not permitted on the node
  (`ERR_NVGPUCTRPERM`).
- At the 1.24 M requests of a 1280 x 720 frame (both fields, sampled majorant; 2.5 M network
  evaluations) the batch costs about 11 ms per sample.

### Cycles with the queue

Same method as [Interactive rendering](#interactive-rendering) (`scripts/srnxr/run_fps_queue.sh`
on p12, `neural_volume_results/fps_queue.csv`): render time of 9 samples minus 1 sample over 8,
CUDA device, CUDA_TENSOR backend, accurate sine, counters off, test scene of
`srnxr_make_scene.py` with both fields (density from `rho`, emission from `p`). Cycles built with
CUDA 12.9 (`sm_120`), the stages run as CUDA kernels on the CUDA device.

| resolution | majorant | queue window | ms per sample | FPS | queue statistics (9 samples) |
|---|---|---|---|---|---|
| 1280 x 720 | sampled | off (inline decode) | 825.0 | 1.21 | |
| 1280 x 720 | sampled | 32 | 44.6 | 22.4 | 16.8 M requests, 30.9 M pairs, 15.99 rows per tile, 109 overflows = 109 replay misses |
| 1280 x 720 | sampled | **128** | **27.9** | **35.9** | 16.8 M requests, 0 overflows, 0 misses |
| 1280 x 720 (density only) | sampled | 32 | 48.0 | 20.8 | 17.1 M requests |
| 1280 x 720 | certified | 32 | 145.4 | 6.9 | 62.4 M requests, 117.4 M pairs, 3578 overflows, 3188 misses |
| 1280 x 720 | certified | **128** | **92.8** | **10.8** | 62.4 M requests, 0 overflows, 0 misses |
| 1280 x 720 (density only) | certified | 32 | 145.5 | 6.9 | 66.0 M requests |

The 15 FPS target at 1280 x 720 is met with the sampled majorant (35.9 FPS, 29.6x the inline
decode); the certified majorant, with 3.7x the requests per sample (9x the mean density), reaches
10.8 FPS.

- The window matters: a lookup the record kernel could not keep (window full) is decoded inline
  at replay by one thread of a nearly empty warp, on the critical path of the launch; 109 of
  17 M lookups cost 17 ms per sample at 720p. The default window is now 128 (0 overflows on the
  test scene with both majorants).
- Per-kernel times of one sample at 1280 x 720, both fields, window 32
  (`scripts/srnxr/run_kstats_queue.sh`): sampled majorant: `neural_queue_tensor` 12.2 ms,
  `integrator_shade_volume` (replay) 8.8 ms, `integrator_neural_record_volume` 6.7 ms,
  `integrator_shade_shadow` 2.1 ms, `integrator_neural_record_shadow` 1.8 ms, expand 1.6 ms,
  scatter 1.2 ms, scan 0.8 ms (5 launches of a 65 536 bucket scan), blend 0.3 ms. Certified
  majorant: tensor 45.6 ms, record 42.0 ms, replay 39.3 ms, shadow 8.2 + 7.0 ms, expand 5.9 ms,
  scatter 4.7 ms: the walks (two per launch) cost as much as the decode; a tighter certified
  bound would cut both.
- Correctness: 960 x 540, both fields, 16 samples, sampled majorant, queue vs inline decode (both
  CUDA_TENSOR): render time 0.374 s vs 8.067 s, mean image value 0.32849 in both, PSNR 67.0 dB
  (noise: the queue decodes every evaluation on Tensor Cores where the inline decoder used the
  SIMT paths for partially active warps, and blends the leaves in neighbor order), largest 16 x 16
  block mean difference 0.0003, 0 replay misses. The accuracy of the decoded field is that of
  CUDA_TENSOR (FP16 activations, [Accuracy](#accuracy)); the batch result matches the direct
  tensor decode to 3.6e-7 (standalone).

### Split FP16 activations (`--neural-tensor-precision split`)

CUDA_TENSOR rounds the activations to FP16 once per layer (the 1.6e-5 NRMSE of the accuracy
table). The queue's tensor stage can instead split every activation into a high FP16 part and
the FP16 rounding of the remainder (`srnxr_split_lo`) and run a second mma stream with the low
parts into the same FP32 accumulator: about 22 bits of the activation reach the product, the
weights stay FP16. Standalone (`srnxr_bench --wavefront --tensor-split`, 16 000 accuracy
points, against the FP32 CUDA_FUSED decode of the same points):

| tensor stage | max abs diff vs FP32, field p | of the field range | tensor stage at 1 M coherent | queue throughput |
|---|---|---|---|---|
| FP16 activations | 2.8e-4 | 9.2e-5 | 7.6 ms | 107.5 Mq/s |
| split (hi + lo) | 2.9e-6 | 9.5e-7 | 10.1 ms | 85.9 Mq/s |

The split variant costs 32 % in the tensor stage (two mma streams, the low parts in local
memory, 255 registers with 92 bytes of spills in the bench build) and brings the activation
error 100x down, to the level of the FP32 decode (the rest is the FP16 rounding of the low part
and the accumulation order).

In Cycles (final build, tensor kernel with its own launch bounds of 256 threads and 255
registers), 1280 x 720, both fields, 1 spp: FP16 activations 30.7 ms (32.5 FPS), split 44.5 ms
(22.5 FPS); certified majorant with split 151.6 ms (6.6 FPS). 960 x 540, 16 samples, sampled
majorant, PSNR against the inline CUDA_FUSED (FP32) render: queue with split 69.5 dB, queue
with FP16 activations 65.4 dB, inline CUDA_TENSOR 68.7 dB; the mean image value is 0.32849 in
all four renders. (At 16 samples the images differ by their Monte Carlo noise, which decorrelates
as soon as the decoded densities change the collision decisions; the PSNR ranks the decoders but
does not measure their error, see the accuracy table for that.)

### Two tiles per warp (not adopted)

`srnxr_v2_eval_network_tensor_block<W, SPLIT, NT>` can evaluate NT tiles per warp (the weight
fragments and the barriers are amortized over 2 x 16 rows, four independent mma chains per
warp). Measured at 1 M coherent requests (`--wf-tiles-per-warp 2`): tensor stage 7.63 ms
against 7.74 ms with one tile (within noise), with split precision 14.6 ms against 10.3 ms
(the second set of fragments and low parts spills, 276 bytes). The kernels are not limited by
the per-row overhead but by the registers; the queue keeps one tile per warp, with its own launch
bounds in Cycles (256 threads, 255 registers: the block evaluator needs 240 - 255 registers, the
168 of the path tracing kernels would spill the fragments).

### Why the certified majorant stays at the leaf level

The certified majorant (9x the mean density on the test model) is the union of per-leaf output
ranges `b +- sum |W_head|` over the leaves touching a voxel. Relative to the true maximum of a
leaf the bound is only 1.4 - 2.0x (median, `srnxr_bounds_check.py`); the rest of the factor is the
ratio of the maximum of a leaf to the mean of a voxel, i.e. the bound is per leaf (16^3 majorant
voxels at depth 3) while the sampled majorant is per voxel. A per-voxel certified bound needs
the range of the network over a sub-box, which interval arithmetic cannot deliver for a SIREN
with w0 = 30: the first-layer pre-activation of a 128^3 voxel spans about 0.9 rad per spatial
axis (|W_0| up to 0.25, w0 = 30, 1/16 of the leaf in local coordinates), so the first sines are
confined, but the second layer sums 256 of them with |w0 W_1| ~ 0.15 and the interval width of
its pre-activations (about 20 rad) already covers full periods; every later layer and the head
are therefore at the global bound. A Lipschitz bound is no better (the constant is about
(30 ||W||)^3 ~ 10^5 in local units). Tighter certified majorants are a property of the model
(smaller leaves where the field varies, or bounds trained and exported with the file), not of the
decoder; the certified mode cannot be tightened in the renderer without changing the certificate.

## Not tested

- RTX 4090 / Ada (SM 8.9), any other GPU than the RTX PRO 6000 Blackwell.
- Real checkpoints of the HPCSE 2026 surrogate; Fourier MLP models (not supported by the decoder).
- CPU rendering of format 2 volumes in Cycles (the CPU path is the host build of CUDA_FUSED, its
  accuracy was tested standalone).
- COOPVEC with OSL shading (rejected by design), on GPUs before Blackwell.
- Profiler counters (Nsight Compute) of Tensor Core utilization and occupancy.
- Cold-cache (first launch after upload) timings per batch; only the first full-set launch was
  recorded (CUDA_TENSOR 11 ms, COOPVEC 20 - 31 ms for 16 000 points).
