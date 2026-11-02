"""Reader and writer of the `.srnxr` neural volume container, format version 2.

Container rules (shared with format version 1):
  * little-endian, an endian marker in the header,
  * fixed binary header followed by a section table, 64-byte aligned sections,
  * `header_crc32` covers header bytes [0, header_crc32_offset),
  * `file_crc32` covers the whole file with the `file_crc32` field zeroed,
  * every section entry carries the CRC-32 of its payload.

Version 2 describes an adaptive spatial octree whose leaves own local SIREN
networks over the 4D query (x, y, z, t). The full mathematical contract is
documented in tools/srnxr/neural_volume.md, the C++ loader is
src/scene/srnxr_model.cpp.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"SRNXR\x00\x01\x00"
FORMAT_VERSION = 2
ENDIAN_MARKER = 0x01020304
ALIGNMENT = 64

DTYPE_FLOAT32 = 1
DTYPE_FLOAT16 = 2
DTYPE_UINT32 = 3
DTYPE_UINT8 = 4
DTYPE_JSON_UTF8 = 5

DTYPE_TO_NUMPY = {
    DTYPE_FLOAT32: np.dtype("<f4"),
    DTYPE_FLOAT16: np.dtype("<f2"),
    DTYPE_UINT32: np.dtype("<u4"),
    DTYPE_UINT8: np.dtype("<u1"),
    DTYPE_JSON_UTF8: np.dtype("<u1"),
}

ARCH_SIREN = 1
ARCH_FOURIER_MLP = 2  # reserved, not supported by the decoders yet

# Header flags
FLAG_LINEAR_HEAD = 1 << 0       # the output layer has no sine activation
FLAG_HAS_FP32_MASTER = 1 << 1   # SIREN_WEIGHTS_FP32 holds the unquantized weights

INVALID_INDEX = 0xFFFFFFFF

SECTION_FIELD_NAMES = "FIELD_NAMES"                    # u8, NUL terminated names
SECTION_LAYER_W0 = "LAYER_W0"                          # f32 [hidden_layer_count]
SECTION_NODES = "NODES"                                # u32 [node_count, 4]
SECTION_NODE_BOUNDS = "NODE_BOUNDS"                    # f32 [node_count, 6]
SECTION_LEAF_NODE = "LEAF_NODE"                        # u32 [leaf_count]
SECTION_LEAF_STRICT_BOUNDS = "LEAF_STRICT_BOUNDS"      # f32 [leaf_count, 8]
SECTION_LEAF_EXPANDED_BOUNDS = "LEAF_EXPANDED_BOUNDS"  # f32 [leaf_count, 8]
SECTION_LEAF_INPUT_TRANSFORM = "LEAF_INPUT_TRANSFORM"  # f32 [leaf_count, 8]
SECTION_LEAF_BLEND = "LEAF_BLEND"                      # f32 [leaf_count, 9]
SECTION_LEAF_NETWORK = "LEAF_NETWORK"                  # u32 [leaf_count]
SECTION_LEAF_OUTPUT_TRANSFORM = "LEAF_OUTPUT_TRANSFORM"  # f32 [leaf_count, 2 * output_dim]
SECTION_NETWORK_LAYOUT = "NETWORK_LAYOUT"              # u32 [network_count, 4]
SECTION_SIREN_WEIGHTS = "SIREN_WEIGHTS"                # f16 or f32
SECTION_SIREN_BIASES = "SIREN_BIASES"                  # f32
SECTION_SIREN_WEIGHTS_FP32 = "SIREN_WEIGHTS_FP32"      # f32, optional master copy
SECTION_MODEL_METADATA_JSON = "MODEL_METADATA_JSON"    # json, optional

NODE_FIELDS = 4          # first_child, leaf_index, parent, depth
NODE_BOUNDS_FIELDS = 6   # min xyz, max xyz
LEAF_BOUNDS_FIELDS = 8   # min xyzt, max xyzt
LEAF_INPUT_FIELDS = 8    # offset xyzt, scale xyzt
LEAF_BLEND_FIELDS = 9    # center xyzt, axis scale xyzt, sigma
NETWORK_LAYOUT_FIELDS = 4  # width, weight_offset, bias_offset, reserved

_HEADER_FORMAT = (
    "<8s"
    + "I" * 3   # header_size, format_version, endian_marker
    + "I" * 10  # architecture, input_dim, output_dim, hidden_layer_count, hidden_width,
                # network_count, leaf_count, node_count, weights_dtype, flags
    + "f" * 8   # domain_min xyzt, domain_max xyzt
    + "Q" * 3   # section_table_offset, section_count, file_size
    + "I" * 2   # header_crc32, file_crc32
)
HEADER_STRUCT_SIZE = struct.calcsize(_HEADER_FORMAT)
HEADER_CRC_FIELD_OFFSET = HEADER_STRUCT_SIZE - 8
FILE_CRC_FIELD_OFFSET = HEADER_STRUCT_SIZE - 4

_SECTION_ENTRY_FORMAT = "<32sIQQQI"
SECTION_ENTRY_SIZE = struct.calcsize(_SECTION_ENTRY_FORMAT)
assert SECTION_ENTRY_SIZE == 64


def align(size: int, alignment: int = ALIGNMENT) -> int:
    remainder = size % alignment
    return size if remainder == 0 else size + (alignment - remainder)


HEADER_BLOCK_SIZE = align(HEADER_STRUCT_SIZE)


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def network_weight_count(input_dim: int, output_dim: int, layers: int, width: int) -> int:
    return width * input_dim + (layers - 1) * width * width + output_dim * width


def network_bias_count(output_dim: int, layers: int, width: int) -> int:
    return layers * width + output_dim


@dataclass
class SrnxrModel:
    """In-memory form of a version 2 file, arrays use the section shapes above."""

    input_dim: int
    output_dim: int
    hidden_layer_count: int
    hidden_width: int
    field_names: list[str]
    layer_w0: np.ndarray
    domain_min: np.ndarray  # [4]
    domain_max: np.ndarray  # [4]
    nodes: np.ndarray  # u32 [node_count, 4]
    node_bounds: np.ndarray  # f32 [node_count, 6]
    leaf_node: np.ndarray  # u32 [leaf_count]
    leaf_strict_bounds: np.ndarray  # f32 [leaf_count, 8]
    leaf_expanded_bounds: np.ndarray  # f32 [leaf_count, 8]
    leaf_input_transform: np.ndarray  # f32 [leaf_count, 8]
    leaf_blend: np.ndarray  # f32 [leaf_count, 9]
    leaf_network: np.ndarray  # u32 [leaf_count]
    leaf_output_transform: np.ndarray  # f32 [leaf_count, 2 * output_dim]
    network_layout: np.ndarray  # u32 [network_count, 4]
    weights: np.ndarray  # f16 or f32, flat
    biases: np.ndarray  # f32, flat
    weights_fp32: np.ndarray | None = None  # optional master copy
    metadata: dict = field(default_factory=dict)
    architecture: int = ARCH_SIREN
    flags: int = FLAG_LINEAR_HEAD

    @property
    def leaf_count(self) -> int:
        return int(self.leaf_node.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def network_count(self) -> int:
        return int(self.network_layout.shape[0])


def _section(name: str, dtype: int, array: np.ndarray) -> tuple[str, int, int, bytes]:
    data = np.ascontiguousarray(array, dtype=DTYPE_TO_NUMPY[dtype]).tobytes()
    return name, dtype, int(np.asarray(array).size), data


def write(path: str, model: SrnxrModel) -> None:
    weights_dtype = DTYPE_FLOAT16 if model.weights.dtype == np.float16 else DTYPE_FLOAT32
    flags = model.flags
    if model.weights_fp32 is not None:
        flags |= FLAG_HAS_FP32_MASTER

    names = b"".join(n.encode("ascii") + b"\x00" for n in model.field_names)
    sections = [
        _section(SECTION_FIELD_NAMES, DTYPE_UINT8, np.frombuffer(names, dtype=np.uint8)),
        _section(SECTION_LAYER_W0, DTYPE_FLOAT32, model.layer_w0),
        _section(SECTION_NODES, DTYPE_UINT32, model.nodes),
        _section(SECTION_NODE_BOUNDS, DTYPE_FLOAT32, model.node_bounds),
        _section(SECTION_LEAF_NODE, DTYPE_UINT32, model.leaf_node),
        _section(SECTION_LEAF_STRICT_BOUNDS, DTYPE_FLOAT32, model.leaf_strict_bounds),
        _section(SECTION_LEAF_EXPANDED_BOUNDS, DTYPE_FLOAT32, model.leaf_expanded_bounds),
        _section(SECTION_LEAF_INPUT_TRANSFORM, DTYPE_FLOAT32, model.leaf_input_transform),
        _section(SECTION_LEAF_BLEND, DTYPE_FLOAT32, model.leaf_blend),
        _section(SECTION_LEAF_NETWORK, DTYPE_UINT32, model.leaf_network),
        _section(SECTION_LEAF_OUTPUT_TRANSFORM, DTYPE_FLOAT32, model.leaf_output_transform),
        _section(SECTION_NETWORK_LAYOUT, DTYPE_UINT32, model.network_layout),
        _section(SECTION_SIREN_WEIGHTS, weights_dtype, model.weights),
        _section(SECTION_SIREN_BIASES, DTYPE_FLOAT32, model.biases),
    ]
    if model.weights_fp32 is not None:
        sections.append(_section(SECTION_SIREN_WEIGHTS_FP32, DTYPE_FLOAT32, model.weights_fp32))
    if model.metadata:
        meta = json.dumps(model.metadata, indent=1, sort_keys=True).encode("utf-8")
        sections.append(
            (SECTION_MODEL_METADATA_JSON, DTYPE_JSON_UTF8, len(meta), meta))

    section_table_offset = HEADER_BLOCK_SIZE
    table_end = section_table_offset + SECTION_ENTRY_SIZE * len(sections)
    entries = []
    offset = align(table_end)
    for name, dtype, count, data in sections:
        entries.append((name, dtype, count, offset, len(data), crc32(data)))
        offset = align(offset + len(data))
    file_size = offset

    payload = bytearray(file_size - table_end)
    for (_, _, _, off, length, _), (_, _, _, data) in zip(entries, sections):
        payload[off - table_end:off - table_end + length] = data

    table = b"".join(
        struct.pack(_SECTION_ENTRY_FORMAT, name.encode("ascii").ljust(32, b"\x00"), dtype, count,
                    off, length, crc)
        for name, dtype, count, off, length, crc in entries)

    def pack_header(header_crc: int, file_crc: int) -> bytes:
        return struct.pack(
            _HEADER_FORMAT, MAGIC, HEADER_STRUCT_SIZE, FORMAT_VERSION, ENDIAN_MARKER,
            model.architecture, model.input_dim, model.output_dim, model.hidden_layer_count,
            model.hidden_width, model.network_count, model.leaf_count, model.node_count,
            weights_dtype, flags, *[float(v) for v in model.domain_min],
            *[float(v) for v in model.domain_max], section_table_offset, len(sections),
            file_size, header_crc, file_crc).ljust(HEADER_BLOCK_SIZE, b"\x00")

    header = pack_header(0, 0)
    header_crc = crc32(header[:HEADER_CRC_FIELD_OFFSET])
    header = pack_header(header_crc, 0)
    blob = header + table + bytes(payload)
    assert len(blob) == file_size
    file_crc = crc32(blob)  # file_crc32 field is still zero here
    blob = pack_header(header_crc, file_crc) + table + bytes(payload)
    with open(path, "wb") as f:
        f.write(blob)


def read(path: str) -> SrnxrModel:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < HEADER_STRUCT_SIZE:
        raise ValueError("truncated .srnxr file")
    fields = struct.unpack(_HEADER_FORMAT, data[:HEADER_STRUCT_SIZE])
    (magic, header_size, version, endian, arch, input_dim, output_dim, layers, width, nets,
     leaves, nodes, weights_dtype, flags) = fields[:14]
    domain_min = np.array(fields[14:18], dtype=np.float32)
    domain_max = np.array(fields[18:22], dtype=np.float32)
    table_offset, section_count, file_size, header_crc, file_crc = fields[22:27]
    if magic != MAGIC:
        raise ValueError("bad magic")
    if endian != ENDIAN_MARKER:
        raise ValueError("endian mismatch")
    if version != FORMAT_VERSION or header_size != HEADER_STRUCT_SIZE:
        raise ValueError(f"unsupported version {version} / header size {header_size}")
    if crc32(data[:HEADER_CRC_FIELD_OFFSET]) != header_crc:
        raise ValueError("header checksum mismatch")
    if file_size != len(data):
        raise ValueError("file size mismatch")
    zeroed = data[:FILE_CRC_FIELD_OFFSET] + b"\x00" * 4 + data[HEADER_STRUCT_SIZE:]
    if crc32(zeroed) != file_crc:
        raise ValueError("file checksum mismatch")

    sections = {}
    for i in range(section_count):
        entry = data[table_offset + i * SECTION_ENTRY_SIZE:table_offset + (i + 1) * SECTION_ENTRY_SIZE]
        name, dtype, count, off, length, crc = struct.unpack(_SECTION_ENTRY_FORMAT, entry)
        name = name.rstrip(b"\x00").decode("ascii")
        payload = data[off:off + length]
        if crc32(payload) != crc:
            raise ValueError(f"section {name} checksum mismatch")
        sections[name] = (dtype, count, payload)

    def arr(name: str, shape=None, optional=False):
        if name not in sections:
            if optional:
                return None
            raise ValueError(f"missing section {name}")
        dtype, count, payload = sections[name]
        a = np.frombuffer(payload, dtype=DTYPE_TO_NUMPY[dtype]).copy()
        if a.size != count:
            raise ValueError(f"section {name} element count mismatch")
        return a.reshape(shape) if shape is not None else a

    names = arr(SECTION_FIELD_NAMES).tobytes().split(b"\x00")[:-1]
    meta = arr(SECTION_MODEL_METADATA_JSON, optional=True)
    return SrnxrModel(
        input_dim=input_dim, output_dim=output_dim, hidden_layer_count=layers, hidden_width=width,
        field_names=[n.decode("ascii") for n in names],
        layer_w0=arr(SECTION_LAYER_W0),
        domain_min=domain_min, domain_max=domain_max,
        nodes=arr(SECTION_NODES, (-1, NODE_FIELDS)),
        node_bounds=arr(SECTION_NODE_BOUNDS, (-1, NODE_BOUNDS_FIELDS)),
        leaf_node=arr(SECTION_LEAF_NODE),
        leaf_strict_bounds=arr(SECTION_LEAF_STRICT_BOUNDS, (-1, LEAF_BOUNDS_FIELDS)),
        leaf_expanded_bounds=arr(SECTION_LEAF_EXPANDED_BOUNDS, (-1, LEAF_BOUNDS_FIELDS)),
        leaf_input_transform=arr(SECTION_LEAF_INPUT_TRANSFORM, (-1, LEAF_INPUT_FIELDS)),
        leaf_blend=arr(SECTION_LEAF_BLEND, (-1, LEAF_BLEND_FIELDS)),
        leaf_network=arr(SECTION_LEAF_NETWORK),
        leaf_output_transform=arr(SECTION_LEAF_OUTPUT_TRANSFORM, (-1, 2 * output_dim)),
        network_layout=arr(SECTION_NETWORK_LAYOUT, (-1, NETWORK_LAYOUT_FIELDS)),
        weights=arr(SECTION_SIREN_WEIGHTS),
        biases=arr(SECTION_SIREN_BIASES).astype(np.float32),
        weights_fp32=arr(SECTION_SIREN_WEIGHTS_FP32, optional=True),
        metadata=json.loads(meta.tobytes().decode("utf-8")) if meta is not None else {},
        architecture=arch,
        flags=flags,
    )
