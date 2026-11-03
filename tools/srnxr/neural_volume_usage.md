# Neural volumes (SRNXR format 2): how to use

A short, practical guide. The design is in [neural_volume.md](neural_volume.md), the
measurements in [neural_volume_results.md](neural_volume_results.md).

## 1. Prerequisites

- Python 3 with `numpy`; `torch` for training the synthetic model (`--fit lstsq` works
  without it, but fits much worse).
- Cycles built from this repository with `WITH_CYCLES_NEURAL_VOLUME=ON` (default) and
  `WITH_CYCLES_NEURAL_CUDA_TENSOR=ON` (default) for a GPU with SM 8.0 or newer
  (`CYCLES_CUDA_BINARIES_ARCH=sm_120` for the RTX PRO 6000 Blackwell). CUDA 12.x; nvcc 13.3
  compiled the kernels in 20+ minutes with 30 - 60 GB of memory, avoid it.
- All commands below run in `tools/srnxr/`.

## 2. Create a model (`.srnxr`)

No trained checkpoints of the CFD surrogate are shipped, so the tests use a synthetic model
fitted to an analytic flow (same layout as the method: adaptive octree, one SIREN per leaf,
15 % overlap, Gaussian blending, fields `ux uy uz p rho` over `(x, y, z, t)`):

```sh
python srnxr_make_synthetic.py synth_w256_l3.srnxr --width 256 --layers 3 --max-depth 3 --threshold 0.5 --steps 3000
```

| option | meaning |
|---|---|
| `--width`, `--layers`, `--w0` | SIREN width (64, 128 or 256 for the Tensor Core paths), number of sine layers, w0 per layer |
| `--max-depth`, `--min-depth`, `--threshold` | octree refinement (refine a node when the density range on a 7^3 probe exceeds the threshold) |
| `--overlap`, `--sigma` | expansion of the leaf boxes and blending width |
| `--fit torch|lstsq`, `--steps`, `--batch`, `--lr` | fitting |
| `--no-master` | do not store the FP32 master weights (smaller file) |

The file format (header, sections, checksums) is implemented in `srnxr_format.py`; a trained
model from another pipeline is written with the same module (`srnxr_format.write`) once its
octree, leaf transforms, blending parameters and weights are available.

## 3. Check the model (optional)

```sh
python srnxr_points.py synth_w256_l3.srnxr points/l3              # test points (accuracy set + workloads)
python srnxr_bounds_check.py synth_w256_l3.srnxr                   # certified per-leaf bounds vs sampling
python srnxr_analysis.py points synth_w256_l3.srnxr points/l3      # slices, gradients, support jumps
```

The standalone decoder bench (CUDA, no Cycles needed) validates and times the GPU decoders:

```sh
cmake -S bench -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build
build/srnxr_bench synth_w256_l3.srnxr points/l3_accuracy.f32 --backend cuda_tensor --out tensor.f32
python srnxr_compare.py synth_w256_l3.srnxr points/l3_accuracy.f32 tensor=tensor.f32   # against the float64 oracle
build/srnxr_bench synth_w256_l3.srnxr points/l3_coherent.f32 --backend cuda_tensor --wavefront --block 256 --batches 65536
build/srnxr_bench synth_w256_l3.srnxr points/l3_coherent.f32 --backend cuda_fused --wavefront --tensor-split   # split precision vs FP32
```

## 4. Create a scene (`.xml` + `.xml.bin`)

```sh
python srnxr_make_scene.py synth_w256_l3.srnxr scene.xml --time 0.5
```

Writes `scene.xml` (a box mesh over the model domain with two volume attributes:
`neural_density` from field `rho`, `pressure` from field `p`; material density =
`max(rho, 0) * 8`, emission = `max(p, 0) * 0.6`; one point light) and the empty `scene.xml.bin`
the XML loader expects. Options: `--time` (simulation time of the frame), `--time-shutter`,
`--density-scale`, `--emission-scale`, `--density-field`, `--emission-field`, `--no-emission`,
`--step` (ray marching step size, 0 = null-collision integrator). The transfer function is the
material: edit the XML to change it, the decoder returns the raw fields.

Per-attribute XML settings (`<attribute volume_type="srnxr" ... >`): `srnxr_backend`,
`srnxr_sine`, `srnxr_fields`, `srnxr_time`, `srnxr_time_shutter`, `srnxr_outside`,
`srnxr_counters`, `srnxr_partial_warps`, `srnxr_majorant`, `srnxr_tensor_precision`
(see neural_volume.md, "Run time selection").

## 5. Render

```sh
cycles --background --device CUDA --width 1280 --height 720 --samples 16 --output out.png \
       --neural-backend cuda_tensor --neural-queue 1 scene.xml
```

Pass `--width/--height` (the standalone ignores the camera resolution of the XML). The
command line overrides the XML:

| option | values | effect |
|---|---|---|
| `--neural-backend` | `cuda_fused` (default), `cuda_tensor`, `coopvec` | decoder: FP32 SIMT, Tensor Cores, OptiX cooperative vectors |
| `--neural-queue` | `0` (default), `1` | wavefront decode queue: record the lookups of the volume walks, decode them in batches on Tensor Cores, replay (29x faster frames; CUDA/OptiX device, not with COOPVEC or ray marching) |
| `--neural-queue-window` | `128` | lookups kept per path and launch; an overflow is decoded inline |
| `--neural-queue-capacity` | `0` = 4 x paths | requests per launch; about 10 GB of device memory at 4.6 M paths |
| `--neural-tensor-precision` | `fp16` (default), `split` | queue tensor stage: FP16 activations, or FP16 high + low parts (FP32-level activations, 1.3x the tensor stage) |
| `--neural-majorant` | `certified` (default), `sampled` | majorant bake from the certified weight bounds (unbiased, 9x the mean density on the test model) or from sampling the field (3x faster, not certified) |
| `--neural-sine` | `accurate` (default), `fast` | `sinf` or `__sinf` |
| `--neural-partial-warps` | `1` (default), `0` | cooperative decoding of partially active warps (inline decoder) |
| `--neural-counters` | `1`, `0` | device statistics of the decoder (off for timing) |
| `--neural-fallback`, `--neural-benchmark` | | backend substitution policy when the device cannot run the requested one |

Interactive numbers on the RTX PRO 6000 Blackwell (1 sample per pixel, both fields):

| configuration | 1280 x 720 |
|---|---|
| inline decoder, sampled majorant | 825 ms (1.2 FPS) |
| queue, sampled majorant | 29 - 31 ms (33 FPS) |
| queue, sampled majorant, split precision | 44.5 ms (22 FPS) |
| queue, certified majorant | 96.5 ms (10 FPS) |

## 6. Reading the output

- `SRNXRImageLoader: Loaded neural volume, backend ...` with the model summary and warnings
  (synthetic model, sampled majorant, FP32 weights rounded to FP16).
- `Neural volume queue: capacity N requests, window K ...` when the queue is allocated.
- At the end: `SRNXR counters (...)` per image (queries, network evaluations per path) and
  `SRNXR queue stats: requests=... overflow=... replay_miss=...`. `overflow` and `replay_miss`
  should be 0; otherwise raise `--neural-queue-window` or `--neural-queue-capacity`.
- Per-kernel GPU times: `CYCLES_DEBUG_PER_KERNEL_PERFORMANCE=1 cycles --log-level trace ...`
  (`integrator_shade_volume`, `integrator_neural_record_volume`, `neural_queue_tensor`, ...).

## 7. Measuring frames per second

Render 1 and N samples in background mode; (time_N - time_1) / (N - 1) is the frame time of
one sample per pixel without scene loading and the majorant bake (the scripts used on p12 are
`scripts/srnxr/run_fps_queue.sh` and `run_kstats_queue.sh` in the project directory there).
