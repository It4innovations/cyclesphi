# The SRNXR file format (`.srnxr`, format version 2)

`.srnxr` stores a **neural volume**: a 3D or 4D field (x, y, z and optionally time t) with up to 8
named output fields (for example `ux uy uz p rho`), compressed as an **adaptive octree of small
local SIREN networks**. Each octree leaf owns one network that is valid in a slightly enlarged
("expanded") box around the leaf. The value at a point is a Gaussian-weighted blend of every leaf
network whose expanded box contains the point.

This document covers:

1. [Quick start: create a `.srnxr` with the example generator](#1-quick-start-create-a-srnxr-with-the-example-generator)
2. [Writing your own model with `srnxr_format.py`](#2-writing-your-own-model-with-srnxr_formatpy)
3. [Binary container](#3-binary-container)
4. [Header](#4-header)
5. [Section table](#5-section-table)
6. [Sections](#6-sections)
7. [Model semantics](#7-model-semantics)
8. [Decoding contract](#8-decoding-contract)
9. [Validation rules of the Cycles loader](#9-validation-rules-of-the-cycles-loader)
10. [File size](#10-file-size)
11. [Related files](#11-related-files)

The Cycles integration (backends, rendering, the wavefront queue) is described in
[neural_volume.md](neural_volume.md) and [neural_volume_usage.md](neural_volume_usage.md).

---

## 1. Quick start: create a `.srnxr` with the example generator

[srnxr_make_synthetic.py](srnxr_make_synthetic.py) builds a complete, valid model. No trained
checkpoints of a real simulation surrogate are shipped, so the generator fits the networks to an
**analytic** time-dependent flow (`analytic_field()` in the script: a vortex flow with a moving
density blob and a steep front). The result has the same layout as a real model, but it is
**synthetic** and the file says so (`"synthetic": true` in its metadata).

### Prerequisites

- Python 3 with `numpy`.
- `torch` (CUDA optional, but much faster) for the default `--fit torch`. Without `torch`, use
  `--fit lstsq`; it runs anywhere but fits much worse (see below).

Run the commands in `tools/srnxr/`. The scripts import each other (`srnxr_format`,
`srnxr_reference`) from that directory.

### Commands

The model used for the Cycles tests (width 256, 3 sine layers, octree depth up to 3):

```sh
cd tools/srnxr
python srnxr_make_synthetic.py synth_w256_l3.srnxr --width 256 --layers 3 --max-depth 3 --threshold 0.5 --steps 3000
```

A quick test without `torch` (a few seconds, poor fit):

```sh
python srnxr_make_synthetic.py test_lstsq.srnxr --fit lstsq --width 64 --layers 3 --max-depth 2
```

Other useful variants:

```sh
# 4 sine layers
python srnxr_make_synthetic.py synth_w256_l4.srnxr --width 256 --layers 4 --threshold 0.5 --steps 3000
# smaller file: no FP32 master weights (about 1/3 of the size)
python srnxr_make_synthetic.py synth_small.srnxr --width 128 --no-master
# a different w0 per sine layer (one value per layer)
python srnxr_make_synthetic.py synth_w0.srnxr --layers 3 --w0 30 10 10
```

### Options

| option | default | meaning |
|---|---|---|
| `output` | (required) | path of the `.srnxr` to write |
| `--width` | 256 | hidden width of every SIREN (the Tensor Core paths are specialized for 64, 128, 256; the maximum is 256) |
| `--layers` | 3 | number of sine (hidden) layers; the output layer is linear and comes on top |
| `--w0` | 30 | SIREN frequency factor: one value for all layers, or one per sine layer |
| `--max-depth` | 3 | maximum octree depth (root = depth 0) |
| `--min-depth` | 1 | the octree is refined uniformly down to this depth |
| `--threshold` | 0.35 | refine a node when the range of `rho` on a 7^3 probe grid at t = 0, 0.5, 1 exceeds this value |
| `--overlap` | 0.15 | expanded box = strict box enlarged by `overlap * size` on both sides of every spatial axis (clipped to the domain) |
| `--sigma` | 0.4 | Gaussian blending width, in units of the strict half extent |
| `--samples` | 6000 | samples per leaf for the output statistics and for `lstsq` |
| `--fit` | `torch` | `torch`: Adam on all leaves at once; `lstsq`: keep the random SIREN sine layers, solve only the linear head by ridge least squares |
| `--steps`, `--batch`, `--lr` | 3000, 1024, 1e-4 | `torch`: Adam steps, samples per leaf and step, learning rate (cosine schedule) |
| `--ridge` | 1e-4 | `lstsq`: ridge factor |
| `--seed` | 1234 | random seed (the `torch` fit is not bitwise reproducible) |
| `--no-master` | off | do not store the `SIREN_WEIGHTS_FP32` section |

### What the script does

1. Builds the adaptive octree over the domain `[-1, 1]^3`, time range `[0, 1]`.
2. For every leaf: the strict box (the octree cell, full time range), the expanded box (15 %
   overlap), the input transform (expanded box -> `[-1, 1]^4`), the blending parameters (center
   and inverse half extent of the strict box, sigma; time does not take part in the blending) and
   the output transform (per-field standard deviation and mean of the target in the expanded box).
3. Initializes one SIREN per leaf (Sitzmann et al. 2020 initialization) and fits it.
4. Stores the weights as FP16 (`SIREN_WEIGHTS`, round to nearest even) and the unrounded FP32
   copy (`SIREN_WEIGHTS_FP32`), writes the file with `srnxr_format.write()`.
5. Reads the file back with `srnxr_format.read()` (all checksums verified) and decodes 20 000
   random held-out points with the float64 reference decoder `srnxr_reference.decode()`.

### Output

Training progress (`--fit torch`) and a JSON report:

```text
step 0 loss 9.869e-01
step 500 loss 1.419e-02
...
step 2999 loss 1.005e-03
{
 "heldout_nrmse_fp16_vs_fp32_master": {"ux": 3.95e-05, "uy": 3.80e-05, "uz": 6.69e-05, "p": 4.60e-05, "rho": 2.05e-05},
 "file": "synth_w256_l3.srnxr",
 "leaves": 120,
 "nodes": 137,
 "max_depth": 3,
 "weights_bytes_fp16": 32010240,
 "fit_nrmse_normalized_mean": [0.0179, 0.0189, 0.0168, 0.0179, 0.0560],
 "heldout_nrmse_vs_analytic": {"ux": 0.0026, "uy": 0.0026, "uz": 0.0044, "p": 0.0031, "rho": 0.0008},
 "seconds": 83.4
}
```

| key | meaning |
|---|---|
| `heldout_nrmse_fp16_vs_fp32_master` | error caused by the FP16 storage of the weights (decode with FP16 vs FP32 weights), normalized by the field range; `null` with `--no-master` |
| `leaves`, `nodes`, `max_depth` | octree size |
| `weights_bytes_fp16` | size of the `SIREN_WEIGHTS` section |
| `fit_nrmse_normalized_mean` | per-field training RMSE in the normalized output space, mean over the leaves |
| `heldout_nrmse_vs_analytic` | RMSE of the blended field against the analytic field on 20 000 random points, divided by the range of each field |
| `seconds` | total run time |

The numbers above are from this machine (torch 2.6 with CUDA): 120 leaves, a 96 MB file (32 MB
FP16 weights + 64 MB FP32 master), 83 s. The `lstsq` quick test above gives 64 leaves,
3.4 MB, 4 s, but a held-out NRMSE of 0.07 - 0.24, about 50 - 100x worse.

### Next steps

```sh
python srnxr_points.py synth_w256_l3.srnxr points/l3            # test point sets
python srnxr_bounds_check.py synth_w256_l3.srnxr                 # certified per-leaf bounds vs sampling
python srnxr_analysis.py points synth_w256_l3.srnxr points/l3    # slices, gradients, support jumps
python srnxr_make_scene.py synth_w256_l3.srnxr scene.xml --time 0.5   # Cycles test scene
```

Rendering is described in [neural_volume_usage.md](neural_volume_usage.md).

---

## 2. Writing your own model with `srnxr_format.py`

A model from another training pipeline is written with the same module. Fill a
`srnxr_format.SrnxrModel` (array shapes as in [section 6](#6-sections)) and call
`srnxr_format.write(path, model)`; the writer computes the header, the section table, the
alignment and all checksums.

A minimal, complete example: one leaf (the octree root), one SIREN with 2 sine layers of width 64,
input (x, y, z, t), one field `density`. Save it as e.g. `minimal_srnxr.py` and run it from
`tools/srnxr/` (or with `tools/srnxr` on `PYTHONPATH`):

```python
"""Minimal hand-built .srnxr: one leaf (the root), one SIREN, 4D input, one field."""
import numpy as np
import srnxr_format as fmt
import srnxr_reference as ref

I, O, L, W = 4, 1, 2, 64                     # input_dim, output_dim, sine layers, width
dmin = np.array([-1, -1, -1, 0], np.float32)  # x y z t
dmax = np.array([1, 1, 1, 1], np.float32)
rng = np.random.default_rng(0)

# Octree with a single node, which is also leaf 0.
nodes = np.array([[fmt.INVALID_INDEX, 0, fmt.INVALID_INDEX, 0]], np.uint32)
node_bounds = np.concatenate([dmin[:3], dmax[:3]])[None, :]
box = np.concatenate([dmin, dmax])[None, :]   # strict = expanded = whole domain

nw = fmt.network_weight_count(I, O, L, W)     # W*I + (L-1)*W*W + O*W
nb = fmt.network_bias_count(O, L, W)          # L*W + O
weights32 = rng.uniform(-0.05, 0.05, nw).astype(np.float32)

model = fmt.SrnxrModel(
    input_dim=I, output_dim=O, hidden_layer_count=L, hidden_width=W,
    field_names=["density"], layer_w0=np.full(L, 30.0, np.float32),
    domain_min=dmin, domain_max=dmax,
    nodes=nodes, node_bounds=node_bounds.astype(np.float32),
    leaf_node=np.array([0], np.uint32),
    leaf_strict_bounds=box, leaf_expanded_bounds=box,
    # x_local = (p - offset) * scale maps the box to [-1, 1]^4
    leaf_input_transform=np.concatenate([0.5 * (dmin + dmax), 2.0 / (dmax - dmin)])[None, :],
    # blend center xyzt, axis scale xyzt, sigma
    leaf_blend=np.array([[0, 0, 0, 0.5, 1, 1, 1, 0, 0.4]], np.float32),
    leaf_network=np.array([0], np.uint32),
    # u = y * scale + offset, per field: [scale..., offset...]
    leaf_output_transform=np.array([[1.0, 0.0]], np.float32),
    network_layout=np.array([[W, 0, 0, 0]], np.uint32),  # width, weight off, bias off, 0
    weights=weights32.astype(np.float16),
    biases=np.zeros(nb, np.float32),
    weights_fp32=weights32,
    metadata={"synthetic": True, "generator": "minimal example"},
)
fmt.write("minimal.srnxr", model)

m = fmt.read("minimal.srnxr")                  # verifies all CRCs
values, inside = ref.decode(m, np.array([[0.1, 0.2, 0.3, 0.5], [2, 0, 0, 0]], np.float32))
print(m.leaf_count, m.network_count, values.ravel(), inside)
```

Output: `1 1 [0.12099402 0.        ] [ True False]` (the second point is outside the domain and
decodes to 0). The file is 29 KB.

Points to get right when exporting a trained model:

- **Weight matrix orientation.** Every matrix is stored `[out][in]`, row-major, which is the
  layout of `torch.nn.Linear.weight`. `W.ravel()` of a `Linear.weight` is correct, `W.T.ravel()`
  is not.
- **Where w0 goes.** A sine layer computes `sin(w0 * (W h + b))`: w0 multiplies the bias too.
  If your network computes `sin(w0 * W h + b)`, store `b / w0` as the bias.
- **Input normalization** belongs in `LEAF_INPUT_TRANSFORM`, **output normalization** in
  `LEAF_OUTPUT_TRANSFORM`, not in the weights (they can be folded in, but then the certified
  bounds of the loader are computed on different values than you expect).
- **The output layer is linear** (no sine). Keep `flags` with `FLAG_LINEAR_HEAD` set
  (the default of `SrnxrModel`).
- **Strict boxes are the octree cells exactly** (bitwise the same floats as `NODE_BOUNDS`), and
  the strict time range of every leaf is the full domain time range.
- Set `"synthetic": false` (or leave the key out) in the metadata of a real model; the Cycles
  loader prints a warning for synthetic ones.

---

## 3. Binary container

```text
offset 0      +--------------------------------------+
              | header (124 bytes, padded to 128)    |
offset 128    +--------------------------------------+
              | section table: section_count x 64 B  |
              +--------------------------------------+  <- 64-byte aligned
              | section payload 0                    |
              +--------------------------------------+  <- 64-byte aligned
              | section payload 1                    |
              | ...                                  |
file_size     +--------------------------------------+  (a multiple of 64)
```

- **Little-endian** everywhere. The header holds the endian marker `0x01020304` (bytes
  `04 03 02 01` on disk); a reader rejects any other value.
- All integers are unsigned, all floats IEEE 754 (FP32, FP16 for weights). No padding between
  header fields or section table fields (Python `struct` with `<`): some 64-bit fields are not
  8-byte aligned.
- Every section payload starts at a multiple of 64 bytes; the gaps are zero. The file size is
  rounded up to a multiple of 64 as well.
- **Checksums** are CRC-32 as in zlib (`zlib.crc32`, polynomial `0xEDB88320` reflected, initial
  value `0xFFFFFFFF`, final XOR `0xFFFFFFFF`):
  - `header_crc32` covers header bytes `[0, 116)` (everything before the field itself).
  - `file_crc32` covers the whole file with the 4 bytes of the `file_crc32` field (offset 120)
    replaced by zeros. It is computed after `header_crc32` has been written.
  - Every section table entry holds the CRC-32 of its payload (`byte_length` bytes).
- Sections are found **by name**, their order does not matter (the writer uses the order of the
  table in [section 6](#6-sections)). Duplicate names are rejected by the Cycles loader; sections
  with unknown names are checksummed and otherwise ignored.

---

## 4. Header

124 bytes (`HEADER_STRUCT_SIZE`), zero padded to 128 (`HEADER_BLOCK_SIZE`). The section table
directly follows at offset 128.

| offset | type | field | value / meaning |
|---:|---|---|---|
| 0 | `u8[8]` | `magic` | `"SRNXR\0\1\0"` = `53 52 4E 58 52 00 01 00` |
| 8 | `u32` | `header_size` | 124 |
| 12 | `u32` | `format_version` | 2 (format 1 is the older 3D uniform-grid layout, see [neural_volume.md](neural_volume.md)) |
| 16 | `u32` | `endian_marker` | `0x01020304` |
| 20 | `u32` | `architecture` | 1 = SIREN; 2 = Fourier MLP (reserved, rejected by the decoders) |
| 24 | `u32` | `input_dim` | 3 (x, y, z) or 4 (x, y, z, t) |
| 28 | `u32` | `output_dim` | number of fields, 1 - 8 |
| 32 | `u32` | `hidden_layer_count` | number of sine layers L, 1 - 8 |
| 36 | `u32` | `hidden_width` | maximum network width, 1 - 256 |
| 40 | `u32` | `network_count` | number of networks (entries of `NETWORK_LAYOUT`) |
| 44 | `u32` | `leaf_count` | number of octree leaves |
| 48 | `u32` | `node_count` | number of octree nodes (internal + leaf) |
| 52 | `u32` | `weights_dtype` | dtype of `SIREN_WEIGHTS`: 2 = FP16, 1 = FP32 |
| 56 | `u32` | `flags` | bit 0 `FLAG_LINEAR_HEAD` (must be set), bit 1 `FLAG_HAS_FP32_MASTER` |
| 60 | `f32[4]` | `domain_min` | x, y, z, t |
| 76 | `f32[4]` | `domain_max` | x, y, z, t |
| 92 | `u64` | `section_table_offset` | 128 |
| 100 | `u64` | `section_count` | number of section table entries |
| 108 | `u64` | `file_size` | total file size in bytes |
| 116 | `u32` | `header_crc32` | CRC-32 of bytes `[0, 116)` |
| 120 | `u32` | `file_crc32` | CRC-32 of the file with this field zeroed |

For a 3D model (`input_dim = 3`) the time components of `domain_min` / `domain_max` are stored
but ignored.

The header of the example model `synth_w256_l3.srnxr`:

```text
magic SRNXR\0\1\0, header_size 124, version 2, endian 0x01020304,
architecture 1, input_dim 4, output_dim 5, hidden_layer_count 3, hidden_width 256,
network_count 120, leaf_count 120, node_count 137, weights_dtype 2 (FP16), flags 3,
domain_min (-1, -1, -1, 0), domain_max (1, 1, 1, 1),
section_table_offset 128, section_count 16, file_size 96433280
```

---

## 5. Section table

`section_count` entries of 64 bytes each, starting at `section_table_offset`:

| offset | type | field | meaning |
|---:|---|---|---|
| 0 | `u8[32]` | `name` | ASCII, NUL padded |
| 32 | `u32` | `dtype` | element type, see below |
| 36 | `u64` | `element_count` | number of elements (not bytes) |
| 44 | `u64` | `byte_offset` | absolute file offset of the payload (multiple of 64) |
| 52 | `u64` | `byte_length` | payload size in bytes, `element_count * sizeof(dtype)` |
| 60 | `u32` | `crc32` | CRC-32 of the payload |

Element types:

| code | name | size |
|---:|---|---:|
| 1 | `DTYPE_FLOAT32` | 4 |
| 2 | `DTYPE_FLOAT16` | 2 |
| 3 | `DTYPE_UINT32` | 4 |
| 4 | `DTYPE_UINT8` | 1 |
| 5 | `DTYPE_JSON_UTF8` | 1 |

The section table of `synth_w256_l3.srnxr` (120 leaves, 137 nodes, 5 fields, L = 3, W = 256):

```text
name                   dtype  element_count  byte_offset  byte_length
FIELD_NAMES            4                 15         1152           15
LAYER_W0               1                  3         1216           12
NODES                  3                548         1280         2192
NODE_BOUNDS            1                822         3520         3288
LEAF_NODE              3                120         6848          480
LEAF_STRICT_BOUNDS     1                960         7360         3840
LEAF_EXPANDED_BOUNDS   1                960        11200         3840
LEAF_INPUT_TRANSFORM   1                960        15040         3840
LEAF_BLEND             1               1080        18880         4320
LEAF_NETWORK           3                120        23232          480
LEAF_OUTPUT_TRANSFORM  1               1200        23744         4800
NETWORK_LAYOUT         3                480        28544         1920
SIREN_WEIGHTS          2           16005120        30464     32010240
SIREN_BIASES           1              92760     32040704       371040
SIREN_WEIGHTS_FP32     1           16005120     32411776     64020480
MODEL_METADATA_JSON    5                991     96432256          991
```

The first payload starts at `align64(128 + 16 * 64) = 1152`.

---

## 6. Sections

Shapes are given as `[rows, columns]`, stored row-major. `N` = `node_count`, `F` = `leaf_count`
(leaves), `K` = `network_count`, `O` = `output_dim`, `L` = `hidden_layer_count`.

| section | dtype | shape | required | content |
|---|---|---|---|---|
| `FIELD_NAMES` | u8 | bytes | yes | `O` unique, non-empty ASCII names, each NUL terminated, e.g. `ux\0uy\0uz\0p\0rho\0` |
| `LAYER_W0` | f32 | `[L]` | yes | w0 of every sine layer (non-zero) |
| `NODES` | u32 | `[N, 4]` | yes | `first_child`, `leaf_index`, `parent`, `depth` |
| `NODE_BOUNDS` | f32 | `[N, 6]` | yes | `min_x, min_y, min_z, max_x, max_y, max_z` |
| `LEAF_NODE` | u32 | `[F]` | yes | octree node index of every leaf |
| `LEAF_STRICT_BOUNDS` | f32 | `[F, 8]` | yes | `min_x, min_y, min_z, min_t, max_x, max_y, max_z, max_t` |
| `LEAF_EXPANDED_BOUNDS` | f32 | `[F, 8]` | yes | same layout; the support of the leaf network |
| `LEAF_INPUT_TRANSFORM` | f32 | `[F, 8]` | yes | `offset_xyzt, scale_xyzt` |
| `LEAF_BLEND` | f32 | `[F, 9]` | yes | `center_xyzt, axis_scale_xyzt, sigma` |
| `LEAF_NETWORK` | u32 | `[F]` | yes | network index of every leaf |
| `LEAF_OUTPUT_TRANSFORM` | f32 | `[F, 2 O]` | yes | `scale[0..O), offset[0..O)` |
| `NETWORK_LAYOUT` | u32 | `[K, 4]` | yes | `width, weight_offset, bias_offset, reserved (0)` |
| `SIREN_WEIGHTS` | f16 or f32 | flat | yes | all weight matrices (dtype = header `weights_dtype`) |
| `SIREN_BIASES` | f32 | flat | yes | all bias vectors |
| `SIREN_WEIGHTS_FP32` | f32 | flat | no | unquantized master copy of `SIREN_WEIGHTS`, same layout; present iff `FLAG_HAS_FP32_MASTER` |
| `MODEL_METADATA_JSON` | json | bytes | no | UTF-8 JSON object, provenance (free form) |

Every float section is read as FP32 or FP16 by the Cycles loader and must contain finite values
only.

### `NODES`, `NODE_BOUNDS`: the octree

- Node 0 is the root: `parent = 0xFFFFFFFF` (`INVALID_INDEX`), `depth = 0`, bounds = the spatial
  part of the domain.
- An **internal node** has `first_child` = index of the first of **8 consecutive** child nodes
  and `leaf_index = INVALID_INDEX`. Child `k` (0 - 7) is the octant selected by the bits of `k`:
  bit 0 = upper half in x, bit 1 = upper half in y, bit 2 = upper half in z. The children split
  the parent exactly at its center `0.5 * (min + max)` (computed in FP32).
- A **leaf node** has `first_child = INVALID_INDEX` and `leaf_index` = its record in the `LEAF_*`
  sections.
- `parent` and `depth` of every child are those of its parent + 1. Maximum depth 32.
- The octree is spatial only (x, y, z); time is never subdivided.

The generator writes the nodes in breadth-first order; any order is valid as long as every node
is reachable from the root exactly once.

### Leaves

For leaf `i`:

- `LEAF_NODE[i]` is its octree node; `NODES[LEAF_NODE[i]].leaf_index == i`.
- **Strict box** `LEAF_STRICT_BOUNDS[i]`: the octree cell (xyz identical to `NODE_BOUNDS`) and the
  full domain time range. The strict boxes partition the domain.
- **Expanded box** `LEAF_EXPANDED_BOUNDS[i]`: contains the strict box; this is where the network
  of the leaf is valid and contributes. The generator enlarges each spatial side by
  `overlap * size` (15 %), clipped to the domain, and keeps the time range.
- **Input transform** `LEAF_INPUT_TRANSFORM[i] = (offset[4], scale[4])`: the network input is
  `x_d = (p_d - offset_d) * scale_d` for the first `input_dim` components. The generator maps the
  expanded box to `[-1, 1]^4`: `offset = center`, `scale = 2 / size`.
- **Blending** `LEAF_BLEND[i] = (c[4], s[4], sigma)`: blending exponent
  `e = -0.5 * sum_d ((p_d - c_d) * s_d)^2 / sigma^2` over d = x, y, z, t. A zero axis scale
  removes that axis from the distance (the generator uses `s_t = 0`, and `s_xyz = 2 / size` of
  the strict box, i.e. the distance is measured in strict half extents). `sigma > 0`, `s >= 0`.
- **Network** `LEAF_NETWORK[i]`: index into `NETWORK_LAYOUT`. Several leaves may share a network.
- **Output transform** `LEAF_OUTPUT_TRANSFORM[i] = (scale[O], offset[O])`: field value
  `u_k = y_k * scale_k + offset_k` from the raw network output `y`. The generator stores the
  standard deviation and mean of each field in the expanded box.

### `NETWORK_LAYOUT`, `SIREN_WEIGHTS`, `SIREN_BIASES`: the networks

Network `n` is described by `NETWORK_LAYOUT[n] = (width, weight_offset, bias_offset, 0)`; the
offsets are **element** offsets into the flat `SIREN_WEIGHTS` / `SIREN_BIASES` arrays (not
bytes). `width` must be `1 .. hidden_width`.

With `I = input_dim`, `W = width`, `O = output_dim`, `L = hidden_layer_count`, the weights of one
network are stored contiguously from `weight_offset`:

| block | shape `[out][in]` | elements |
|---|---|---:|
| sine layer 0 | `[W][I]` | `W * I` |
| sine layers 1 .. L-1 | `[W][W]` each | `(L - 1) * W * W` |
| output (head) layer | `[O][W]` | `O * W` |

and the biases from `bias_offset`:

| block | elements |
|---|---:|
| sine layers 0 .. L-1 | `W` each, `L * W` total |
| output layer | `O` |

`srnxr_format.network_weight_count()` and `network_bias_count()` compute these counts. Each matrix
is row-major with one row per output neuron: element `(r, c)` of layer `l` is at
`weight_offset + (offset of layer l) + r * in + c`.

Example (`synth_w256_l3.srnxr`, I = 4, W = 256, L = 3, O = 5): `256*4 + 2*256*256 + 5*256 =
133376` weights and `3*256 + 5 = 773` biases per network, so
`NETWORK_LAYOUT[1] = (256, 133376, 773, 0)`.

The Cycles loader always deploys the weights as FP16: FP32 weights in `SIREN_WEIGHTS` are rounded
to nearest even on load (and reported), a weight that overflows FP16 is an error. Biases stay
FP32. `SIREN_WEIGHTS_FP32` is not used by Cycles; it serves the Python tools (accuracy of the FP16
storage, `srnxr_reference.decode(..., use_master=True)`).

### `MODEL_METADATA_JSON`

Free-form UTF-8 JSON object. The Cycles loader only looks for `"synthetic": true` and prints a
warning when it is present. The generator stores:

| key | content |
|---|---|
| `synthetic` | `true` |
| `generator` | `tools/srnxr/srnxr_make_synthetic.py` |
| `args` | the command line options |
| `fit` | description of the fitting method |
| `field_definition` | where the analytic target is defined |
| `time_range` | `[t_min, t_max]` |
| `overlap_convention`, `blend_convention` | text description of the leaf box and blending conventions |

---

## 7. Model semantics

```text
          domain (x, y, z), time range [t0, t1] not subdivided
+-------------------+---------+---------+
|                   |  leaf   |  leaf   |
|                   |    B    |    C    |      strict boxes: the octree cells, a partition
|      leaf A       +----+----+---------+
|                   | D  | E  |         |      expanded boxes: strict box + 15 % on each side,
|                   +----+----+  leaf F |      where the leaf network is valid
|                   | G  | H  |         |
+-------------------+----+----+---------+

value at p = Gaussian-weighted mean of every leaf network whose expanded box contains p
```

Per leaf `j`, a SIREN with L sine layers and a linear head:

```text
x   = (p - offset_j) * scale_j                     local input, input_dim components
h_0 = x
h_l = sin(w0_l * (W_l h_(l-1) + b_l))              l = 1 .. L    (w0_l = LAYER_W0[l - 1])
y   = W_head h_L + b_head                          raw output, output_dim components
u_j = y * out_scale_j + out_offset_j               field values
e_j = -0.5 * |(p - c_j) * s_j|^2 / sigma_j^2       blending exponent
```

The weights in the file are those of the deployed network; nothing else (activation choice,
normalization, encoding) is implied.

---

## 8. Decoding contract

Implemented by the float64 oracle [srnxr_reference.py](srnxr_reference.py) (`decode()`) and by
the Cycles device decoder `srnxr_v2_decode()` (`src/kernel/util/image_3d_srnxr_decode.h`):

1. Query `p = (x, y, z, t)`; for a 3D model t is ignored. In Cycles
   `t = time_frame + time_shutter * (ray_time - 0.5)`.
2. If p is outside the **closed** domain box, every field is `outside_value` (0 by default) and
   nothing is evaluated.
3. **Strict leaf** S: descend the octree from the root; at each internal node pick child
   `k = (x >= cx) | (y >= cy) << 1 | (z >= cz) << 2` with the FP32 center `c` of the node. A
   coordinate equal to the center goes to the upper child.
4. **Neighbors** of S: every leaf j whose expanded box intersects the strict box of S (closed
   intervals, xyz and, for 4D models, t), in ascending leaf index. The strict leaf itself is
   always one of them. The lists are computed by the loader; they are not stored in the file.
5. For every neighbor j with p in the **closed** expanded box of j: evaluate `u_j` and `e_j`
   ([section 7](#7-model-semantics)).
6. Blend with the max shift `m = max_j e_j`:

   ```text
   u(p) = sum_j exp(e_j - m) * u_j  /  sum_j exp(e_j - m)
   ```

   The largest weight is exactly 1, so the weight sum is >= 1 and never underflows; since the
   strict leaf always contributes, there is no empty-sum case inside the domain.

The truncation of the Gaussian weights at the expanded boxes is part of the method (it creates a
small discontinuity at the box faces; `srnxr_analysis.py` measures it as "support jump"). The
decoded fields are raw values with no clamping; mapping them to density, emission or color is the
job of the material.

The certified per-leaf output range used for the majorants in Cycles follows from `|sin| <= 1`:
`y_k` lies in `b_head,k +- sum_c |W_head,kc|` for any input
(`srnxr_reference.certified_leaf_bounds()`, `srnxr_bounds_check.py`).

---

## 9. Validation rules of the Cycles loader

`src/scene/srnxr_model.cpp` rejects a file unless all of the following hold (the Python
`srnxr_format.read()` checks only the container: magic, endian, version, header size, file size
and the three kinds of checksums, plus element counts):

Container
- magic, endian marker, `format_version == 2`, `header_size == 124`;
- header CRC, file size (`file_size` = actual size, catches truncation), file CRC;
- section table and every payload inside the file, every section CRC, no duplicate names;
- all required sections present with the expected element counts and dtypes;
  `byte_length == element_count * sizeof(dtype)`.

Header
- `architecture == 1` (SIREN), `FLAG_LINEAR_HEAD` set;
- `input_dim` 3 or 4, `output_dim` 1 - 8, `hidden_layer_count` 1 - 8, `hidden_width` 1 - 256;
- `network_count`, `leaf_count`, `node_count` > 0; `weights_dtype` FP16 or FP32;
- finite domain with `max > min` on every used axis.

Content
- `FIELD_NAMES` holds exactly `output_dim` unique, non-empty, NUL terminated names;
- no zero in `LAYER_W0`; all float values finite;
- octree: node 0 is the root with the domain bounds; every node reached exactly once; no zero
  extent; depth <= 32; internal nodes have 8 in-range children with the right parent and depth,
  whose bounds are the octants of the parent (relative tolerance 1e-6); leaf nodes own exactly one
  leaf record that points back to them;
- leaves: strict box == node box (exact), strict time range == domain time range (time-partitioned
  leaves are not supported), expanded box contains the strict box, non-zero input scales,
  `sigma > 0`, axis scales >= 0, valid network index;
- networks: `1 <= width <= hidden_width`, weights and biases inside their arrays, both arrays
  below 2^32 elements;
- deployed FP16 weights finite.

The loader also selects the fields of an image (`srnxr_fields`: one or three names; default the
only field, or `rho` / `density`) and reports the measured overlap, the octree depth and the
neighbor counts.

---

## 10. File size

Approximately

```text
size ~ K * (W*I + (L-1)*W^2 + O*W) * (2 bytes FP16 + 4 bytes FP32 master, if present)
     + K * (L*W + O) * 4                      biases
     + F * (8+8+8+9+2O) * 4 + F * 8           leaf records
     + N * (4*4 + 6*4)                        octree
     + small header, table, names, metadata
```

The weights dominate: for W = 256, L = 3 one network is 133 376 weights = 261 KB in FP16
(+ 521 KB FP32 master). `synth_w256_l3.srnxr` with 120 networks is 96 MB, or 32 MB with
`--no-master`.

---

## 11. Related files

| file | purpose |
|---|---|
| [srnxr_format.py](srnxr_format.py) | reader and writer, all constants of this document |
| [srnxr_reference.py](srnxr_reference.py) | float64 reference decoder, certified bounds |
| [srnxr_make_synthetic.py](srnxr_make_synthetic.py) | example generator (this document, section 1) |
| [srnxr_points.py](srnxr_points.py), [srnxr_compare.py](srnxr_compare.py), [srnxr_analysis.py](srnxr_analysis.py), [srnxr_bounds_check.py](srnxr_bounds_check.py), [srnxr_majorant_check.py](srnxr_majorant_check.py), [srnxr_report.py](srnxr_report.py) | test points, decoder comparison, field analysis, bound checks, report |
| [srnxr_make_scene.py](srnxr_make_scene.py) | Cycles XML test scene for a model |
| [bench/](bench/) | standalone CUDA / OptiX decoder benchmark |
| `src/scene/srnxr_model.{h,cpp}` | C++ parser, validator, device packer |
| `src/kernel/util/image_3d_srnxr*.h` | device structures and decoders |
| [neural_volume.md](neural_volume.md) | design of the Cycles integration |
| [neural_volume_usage.md](neural_volume_usage.md) | how to build, render and measure |
| [neural_volume_results.md](neural_volume_results.md) | measurements |
